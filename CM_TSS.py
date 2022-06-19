import json
import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize


class CM_TSS():
    def __init__(self, settings_dir: Path) -> None:
        """
        Object to run Cerjan-Miller saddle point search calculations given a settings file.
        """
        # Make sure settings_dir is a Path
        if not isinstance(settings_dir, Path):
            settings_dir = Path(settings_dir)
        # Read the setting from the setting.json file
        with open(settings_dir) as f:
            settings_dict_in = json.load(f)
        
        # Turn the setting dict into a default dict to prevent exceptions
        settings_dict = defaultdict(str)
        for key, val in settings_dict_in.items():
            settings_dict[key] = val
        
        self.N = settings_dict['N']
        self.D = settings_dict['Dim'] if not (settings_dict['Dim'] == '') else 3
        self.charge = settings_dict['charge'] if not (settings_dict['charge'] == '') else 0
        self.spin = settings_dict['spin'] if not (settings_dict['spin'] == '') else 1
        self.N_procs = settings_dict['N-procs'] if not (settings_dict['N-procs'] == '') else 8
        self.R_conv = settings_dict['conv-radius'] if not (settings_dict['conv-radius'] == '') else 0.1
        self.G_conv = settings_dict['conv-grad'] if not (settings_dict['conv-grad'] == '') else 1e-6
        self.max_iter = settings_dict['max-iter'] if not (settings_dict['max-iter'] == '') else 10
        self.R_trust = settings_dict['R-trust'] if not (settings_dict['R-trust'] == '') else 0.1
               
        self.init_structure = Path(settings_dict['working-dir']) / (settings_dict['init-f-name'])
        if self.init_structure == "":
            raise RuntimeError("No initial structure specified. Check setting.json file.")

        self.basis_dir = Path(settings_dict['working-dir']) / (settings_dict['basis-f-name']) if not (settings_dict['basis-f-name'] == '') else ''
        self.hist_file = Path(settings_dict['working-dir']) / ((settings_dict['history-f-name'] + '.xyz') if not (settings_dict['history-f-name'] == '') else 'history.xyz')
        self.final_file = Path(settings_dict['working-dir']) / ((settings_dict['final-f-name'] + '.xyz') if not (settings_dict['final-f-name'] == '') else 'final.xyz')
        self.gjf_dir = Path(settings_dict['working-dir']) / ((settings_dict['gaussian-f-name'] + '.gjf') if not (settings_dict['gaussian-f-name'] == '') else 'in.gjf')
        self.log_dir = Path(settings_dict['working-dir']) / ((settings_dict['gaussian-f-name'] + '.log') if not (settings_dict['gaussian-f-name'] == '') else 'in.log')
        self.chk_dir = Path(settings_dict['working-dir']) / ((settings_dict['gaussian-f-name'] + '.chk') if not (settings_dict['gaussian-f-name'] == '') else 'in.chk')
        self.fchk_dir = Path(settings_dict['working-dir']) / ((settings_dict['gaussian-f-name'] + '.fchk') if not (settings_dict['gaussian-f-name'] == '') else 'in.fchk')
        self.submit_dir = settings_dict['submit-f-dir']

        self.energy_calc_header = settings_dict['force-header-calc'] if not (settings_dict['force-header-calc'] == '') else "#P wB97XD/6-31G** nosymm force" 
        self.hess_calc_header = settings_dict['hess-header-calc'] if not (settings_dict['hess-header-calc'] == '') else "#P wB97XD/6-31G** nosymm freq"

        self.init_coords, self.atom_types, self.periphery = self._read_coords()

        self.num_moving_atoms = self.N - (-sum(self.periphery))

        self.atom_dict = {'H': 1, 'O': 8, 'Al':13, 'F': 9, 'C': 6, 'N': 7}
        self.atom_dict_r = {1: 'H', 8: 'O', 13: 'Al', 9: 'F', 6: 'C', 7: 'N'}
        self.atom_types_name = [self.atom_dict_r[i] for i in self.atom_types]
        

        self.E = 0
        self.H = np.zeros((self.D*self.N, self.D*self.N))
        self.G = np.zeros((1, self.D*self.N))
        self.G_hist = np.zeros((self.max_iter, ))
        self.dx = np.zeros((1, self.D*self.N))
        self.evec_mem = []
    
    def _read_coords(self) -> tuple:
        """
        Reads geometric data from .inp file with exactly self.N lines with format:
        atom_type   periphery(0 or -1)   x   y   z.
        Returns:
            init_coords (np.array): initial coordinates.
            atom_types (np.array): atom types.
            periphery (np.array): peripheral atoms' indecies
        """
        init_coords = np.zeros((self.N, self.D))
        periphery = np.zeros((self.N, ), dtype='int8')
        atom_types = np.zeros((self.N, ), dtype='int8')
        atom_ind = 0
        with open(self.init_structure) as f:
            for line in f:
                line_s = line.split()
                atom_types[atom_ind] = line_s[0]
                periphery[atom_ind] = line_s[1]

                for d in range(self.D):
                    init_coords[atom_ind, d] = line_s[2+d]

                atom_ind +=1
            
        return init_coords, atom_types, periphery

    def _get_priphery_H(self) -> np.ndarray:
        """
        Returns the hessian of the moving atoms only so it's invertible and non-zero
        """
        peri_H = np.copy(self.H)
        peri_H = peri_H[~np.all(peri_H == 0, axis=1)]
        peri_H = peri_H.T[~np.all(peri_H == 0, axis=0)]
        return peri_H

    def _get_padded_dx(self, dx: np.ndarray) -> np.ndarray:
        """Returns the padded dx vector

        Args:
            dx (np.array): dx for moving atoms: (self.D*self.num_moving_atoms, )

        Returns:
            padded_dx (np.array): dx padded with zeros for frozen atoms: (self.D*self.N, )
        """
        cnt = 0
        padded_dx = np.zeros((self.N*self.D, ))
        for i in range(len(self.periphery)):
            if self.periphery[i] != -1:
                for d in range(self.D):
                    padded_dx[self.D*i+d] = dx[cnt+d]
    
                cnt += self.D
        
        return padded_dx

    def _get_grad(self, inplace: bool =True) -> np.ndarray:
        """Returns the gradient vector

        Args:
            inplace (bool): if True, sets self.G equal to the result, otherwise, returns the array.

        Returns:
            G_out (np.array): gradient vector: (self.D*self.N, )
        """
        with open(self.fchk_dir) as f:
            f_cnt = f.readlines()
            for ind, line in enumerate(f_cnt):
                line_s = line.split()
                if all(i in line_s for i in ['Cartesian', 'Gradient']):
                    start_ind = ind+1
                    break
            
            if (self.D*self.N)%5 == 0:
                end_ind = start_ind + (self.D*self.N)//5
            else:
                end_ind = start_ind + (self.D*self.N)//5 + 1

            G_raw = f_cnt[start_ind:end_ind]

        G_ind = 0
        G_out = np.zeros_like(self.G)
        for line in G_raw:
            line_list = line.split()
            for num in line_list:
                G_out[0, G_ind] = float(num)
                G_ind += 1
        
        if inplace:
            self.G = G_out
        else:
            return G_out
    
    def _get_hessian(self) -> None:
        """ Calculates Hessian. Sets the self.H variable.
        """
        with open(self.fchk_dir) as f:
            f_cnt = f.readlines()
            for ind, line in enumerate(f_cnt):
                line_s = line.split()
                if all(i in line_s for i in ['Cartesian', 'Force', 'Constants']):
                    start_ind = ind+1
                    break

            H_tot_size = int(self.D*self.N * (self.D*self.N + 1) / 2)
            if H_tot_size%5 == 0:
                end_ind = start_ind + H_tot_size//5
            else:
                end_ind = start_ind + H_tot_size//5 + 1

            H_raw = f_cnt[start_ind:end_ind]

        H_list = []
        for line in H_raw:
            line_list = line.split()
            for num in line_list:
                H_list.append(float(num))
                
        list_cntr = 0
        for i in range(self.D*self.N):
            for j in range(0, i+1):
                self.H[i, j] = H_list[list_cntr]
                self.H[j, i] = H_list[list_cntr]
                list_cntr += 1

        return

    def _get_energy(self, inplace: bool =True) -> float:
        """
        Get SCF energy from fchk file.
        Args:
            inplace (bool): if True, sets self.E equal to the result, otherwise, returns the value.

        Returns:
            E (float): SCF energy value in Hartrees
        """
        with open(self.fchk_dir) as f:
            f_cnt = f.readlines()
            for ind, line in enumerate(f_cnt):
                line_s = line.split()
                if all(i in line_s for i in ['SCF', 'Energy']):
                    line_ind = ind
                    break
            
            E_list = f_cnt[line_ind].split()

        if inplace:
            self.E = float(E_list[-1])
            return
        else:
            return float(E_list[-1])

    def _sub_gaussian(self) -> bool:
        """
        Submit a gaussian job for input file self.gjf_dir.
        Returns:
            status (bool): 0 for successful and 1 for failed jobs.
        """
        os.system("g16 {} {}".format(self.gjf_dir, self.log_dir))
        with open(self.log_dir) as f:
            f_cnt = f.readlines()
            f_cnt = f_cnt[::-1]
            f_cnt = f_cnt[:100]
            for line in f_cnt:
                if all(i in line.split() for i in ['Normal', 'termination']):
                    return 0
            return 1

    def _write_gaussian(self, struct: np.ndarray, c_type: str='H') -> None:
        """
        write a freq calculation g16 input file with struct coords and self.basis for basis.
        Args:
            struct (np.ndarray): coordinates to write to input file: (self.D*self.N, )
            c_type (str): calculation type. 'H' is for hessian calculations. Otherwise, it just calculates the energy.
        """
        str_list = ["%NProcShared={}\n".format(self.N_procs),
        "%chk={}\n".format(self.chk_dir),
        "{}\n".format(self.hess_calc_header if c_type == 'H' else self.energy_calc_header),
        "\n",
        "EF-TSS-calc-{}\n".format(c_type),
        "\n",
        "{} {}\n".format(self.charge, self.spin)
        ]

        for i in range(self.N):
            str_list.append("{}\t{}\t{}\t{}\t{}\n". format(self.atom_types[i], self.periphery[i], struct[i][0], struct[i][1], struct[i][2]))
        
        str_list.append("\n")

        if self.basis_dir != '':
            with open(self.basis_dir) as f:
                basis_list = f.readlines()

            str_list += basis_list
        
        str_list.append("\n")
        str_list.append("\n")

        with open(self.gjf_dir, 'w') as f:
            f.writelines(str_list)

        return

    def _write_history(self, struct: np.ndarray, fname: str) -> None:
        """
        write struct to .xyz file.
        Args:
            struct (np.ndarray): coordinates to write to .xyz file: (self.D*self.N, )
            fname (str): name of the .xyz file.
        """
        str_list = []
        for i in range(self.N):
            coord_str = "{}\t".format(struct[i, 0])
            for d in range(self.D-1):
                coord_str += "{}\t".format(struct[i, d+1])

            coord_str = "{}\t".format(self.atom_types_name[i]) + coord_str + "\n"
            str_list.append(coord_str)
        
        str_list.insert(0, "\n")
        str_list.insert(0, "{}\n".format(self.N))
        
        with open(fname, 'a') as f:
            f.writelines(str_list)

        return

    def _Muller_Brown(self, x: np.ndarray) -> tuple:
        """Muller_Brown surface for testing purposes.
        Args:
            x (np.ndarray): input structure: (self.D*self.N, )
        Returns:
            f (float): free energy
            g (np.ndarray): gradient of f: (self.D*self.N, )
            h(np.ndarray): Hessian of f: (self.D*self.N, self.D*self.N)
        """
        x = x.squeeze()
        A = [-200, -100, -170, 15]
        a = [-1, -1, -6.5, 0.7]
        b = [0, 0, 11, 0.6]
        c = [-10, -10, -6.5, 0.7]
        x0 = [1, 0, -0.5, -1.0]
        y0 = [0, 0.5, 1.5, 1.0]
        func = lambda i: a[i]*(x[0]-x0[i])**2 + b[i]*(x[0]-x0[i])*(x[1]-y0[i]) + c[i]*(x[1]-y0[i])**2
        func_x = lambda i: 2*a[i]*(x[0]-x0[i]) + b[i]*(x[1]-y0[i])
        func_y = lambda i: 2*c[i]*(x[1]-y0[i]) + b[i]*(x[0]-x0[i])
        func_xy = lambda i: b[i]
        func_xx = lambda i: 2*a[i]
        func_yy = lambda i: 2*c[i]

        f = sum([A[i]*np.exp(func(i)) for i in range(4)])
        g = [sum([A[i]*func_x(i)*np.exp(func(i)) for i in range(4)]),
             sum([A[i]*func_y(i)*np.exp(func(i)) for i in range(4)])]
        h = [[sum([A[i]*np.exp(func(i))*(func_xx(i) + func_x(i)*func_x(i)) for i in range(4)]),
        sum([A[i]*np.exp(func(i))*(func_xy(i) + func_x(i)*func_y(i)) for i in range(4)])],
        [sum([A[i]*np.exp(func(i))*(func_xy(i) + func_x(i)*func_y(i)) for i in range(4)]),
        sum([A[i]*np.exp(func(i))*(func_yy(i) + func_y(i)*func_y(i)) for i in range(4)])]]

        return f, np.array(g), np.array(h)

    def run(self) -> None:
        """
        Run CM_TSS (Cerjan-Miller Transition State Search) algorithm with the initialized structure and parameters. The algorithm switches to Newton-Raphson after 
        crossing the inflection point.
        """

        curr_x = self.init_coords
        iter = 0
        ask_flag = True

        while True:
            self._write_gaussian(curr_x)
            fail_flag = self._sub_gaussian()
            if fail_flag:
                    raise RuntimeError("Initial force calculations failed. Check {} file".format(self.log_dir))
            os.system("formchk {} {} > /dev/null 2>&1".format(self.chk_dir, self.fchk_dir))
            self._get_energy()
            self._get_grad()
            self._get_hessian()

            peri_H = self._get_priphery_H()
            peri_G = self.G[self.G != 0]
            evals, U = np.linalg.eig(peri_H) ## U^T H U = evals
            gamma = U.T @ peri_G.T
            gamma = gamma.reshape((self.D*self.num_moving_atoms, 1))
            
            
            if np.all(evals >= 0):
                # If all eigenvalues are positive, continue the CM_TSS algorithm
                step_type = "CM"
                
                mode = 0 # Automatically, CM method chooses the smallest eigenvalue to follow. This is not neccessarily correct...
                if ask_flag:
                    ask = input("which mode to follow: ")
                else:
                    ask = None
                
                if ask == "":
                    ask_flag = False
                elif not(ask is None):
                    try:
                        mode = int(ask)
                    except:
                        raise ValueError("Invalid mode selected. Terminating.")


                del_V = lambda l: (gamma.T @ ((l*np.ones_like(peri_H) - np.diag(evals)/2)/(l*np.ones_like(peri_H) - np.diag(evals))**2) @ gamma).squeeze()

                res = minimize(del_V, sum(sorted(evals)[mode:mode+2])/2)
                l0 = res.x[0]

                dx = U@np.linalg.inv(l0*np.ones_like(peri_H) - np.diag(evals))@gamma
            
            else:
                # switch to Newton-Raphson
                step_type = "NR"
                ksi = -np.linalg.inv(np.diag(evals))@gamma
                dx = U@ksi

            
            self.dx = self._get_padded_dx(dx)
            iter_dx_size = np.linalg.norm(dx)

            print("Iteration {}({}):\tdx: {}\tgrad: {}". format(iter, step_type, iter_dx_size, np.linalg.norm(self.G)))


            # Resize if bigger than trust radius
            #if step_type == "NR":
            if np.linalg.norm(self.dx) > self.R_trust:
                self.dx = self.R_trust * self.dx/np.linalg.norm(self.dx)

            # write current geometry to history file
            self._write_history(curr_x, self.hist_file)

            # update geometry
            curr_x += self.dx.reshape(self.N, self.D)
            
            # Check conversion criteria
            if np.linalg.norm(iter_dx_size) <= self.R_conv:
                print("R_conv satisfied. Writing the final structure to final.xyz")
                break
            
            if np.linalg.norm(self.G) <= self.G_conv:
                print("G_conv satisfied. Writing the final structure to final.xyz")
                break
            
            if iter >= self.max_iter:
                print("Max_iter reached. Writing the final structure to final.xyz")
                break
            
            self.G_hist[iter] = np.linalg.norm(self.G)
            
            iter += 1
                

        
        self._write_history(curr_x, self.final_file)

        return





