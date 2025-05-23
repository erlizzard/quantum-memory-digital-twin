import numpy as np 
from scipy.special import gamma, rgamma
import matplotlib.pyplot as plt
from scipy.special import factorial
import json

#with open('nb_scan\\nb_fit_paramters.txt', 'r') as file:
      #txt =  file.read()
#       nb_dict  = json.load(file)
#nb_dict = {}
#class_names = ['LambdaCsD1','FLAME2_Rb', 'ORCA_Cs_D2', 'Lambda_Rb_D1', 'FLAME_Cs_D1', 'LambdaCsD1Compact','ORCA_Telecom','LambdaRbD1Compact','SuperradianceRbD2',]
#for i in range(len(class_names)):
#    nb_dict[class_names[i]]  = 0

class Memory():
    def __init__(self,int_eff, transmission,nB, input_dim,internal_dim):
        self.int_eff = int_eff
        self.transmission = transmission

        self.t_in = 1-np.sqrt(self.int_eff)
        self.t_out = 1-np.sqrt(self.int_eff)
        self.eta_e = self.transmission
        self.eta_l = self.transmission
        self.nB_e = nB
        self.nB_l = nB

        self.input_dim = input_dim
        self.internal_dim = internal_dim
        self.n_max = self.input_dim-1


    def homemade_Jacobi(self,n,a,b,x):
        '''Scipy jacobi uses recurrence, which breaks for n = 0, so we implement our own. only works for real numbers.'''
        result = 0
        for s in range(0,n+1):
            result += (gamma(n+a+1)*(rgamma(n-s+1)*rgamma(a+s+1))) * (gamma(n+b+1)*(rgamma(s+1)*rgamma(n+b-s+1))) * (((x-1)/2)**s) * (((x+1)/2)**(n-s))
            
        return result
    
    def get_bs_matrix_element(self, N,n,m,t):

        #this uses the mathematica simplify and I don't think it is right? 
        r = np.sqrt(1-t**2)
        #print(n, m-n, N-n-m, t**2-r**2)

        a = np.sqrt(factorial(N-n)*factorial(n))/ np.sqrt(factorial(N-m)*factorial(m))

        if t ==0: 
            t = 1e-16

        b = t**(N-2*n) * (r/t)**(m-n) 

        if b == np.inf:
            b = 1
    
        p = self.homemade_Jacobi(n, (m-n), (N-n-m), t**2-r**2)
        return a*b*p
    
    def make_beamsplitter_U(self,t, op_idx):
        r = np.sqrt(1- t**2)
        full = 0

        if op_idx == 0 :
            e_dim = self.input_dim
            l_dim = self.internal_dim
        if op_idx == 1:
            e_dim = self.internal_dim
            l_dim = self.input_dim

        total_dim = e_dim + l_dim

        for N in range(0,self.n_max+1):
            for n in range(0,N+1):
                for m in range(0,N+1):
        
                    input_state_label = (N-n,n)
                    output_state_label = (N-m,m)
        
                    #print('input: ', input_state_label)
                    #print('output: ', output_state_label)

                    state_early_in = np.zeros(e_dim).reshape(-1,1)
                    state_early_in[input_state_label[0]] = 1
                    
                    state_late_in = np.zeros(l_dim).reshape(-1,1)
                    state_late_in[input_state_label[1]] = 1

                    joint_in = np.kron(state_early_in,state_late_in)
    
                    state_early_out = np.zeros(e_dim).reshape(-1,1)
                    state_early_out[output_state_label[0]] = 1

                    state_late_out = np.zeros(l_dim).reshape(-1,1)
                    state_late_out[output_state_label[1]] = 1
                    
                    joint_out = np.kron(state_early_out,state_late_out)
                    
                    full_state = np.outer(joint_out,joint_in)
        
                    U_elem = self.get_bs_matrix_element(N,n,m,t)
                    full += np.multiply(U_elem, full_state)
        return full
    
    def make_A_joint(self,eta, n_b):

        g = 1 + (1-eta)*n_b
        tau = eta/g
        #max photon number
    
        a  = np.zeros((self.input_dim, self.input_dim))
        for i in range(0,self.input_dim-1):
            a[i,i+1] = np.sqrt(i+1)
        a_dagger = a.conj().T
        n_hat = np.diag(a_dagger@a )


        A_list = []
        #l is number of lost photons
        for l in range(self.input_dim):
            prefactor = np.sqrt((1-tau)**l/(factorial(l)))
            A_list.append(prefactor * np.diag(tau**(n_hat/2)) @ np.linalg.matrix_power(a,l))

       
        return A_list 
    
    def make_B_joint(self,eta, n_b):
        """
        Constructs the Kraus matrix representing the gain of photons in the system $B_j$
        
        Args:
            eta (float): Transmissivity of the beam splitter coupling to the environment
            n_b (float): Mean thermal photon number.
            op_idx (int): gives the index of the state we create the Kraus matrix for.
            state_dim (int): dimension of the state we should create the Kraus matrix for


        Returns:
            list of np.ndarray: returns the Kraus operators applied on the space given by the state index
        """
        g = 1 + (1-eta)*n_b

        
        a_dagger  = np.zeros((self.input_dim, self.input_dim ))
        for i in range(self.input_dim-1):
            a_dagger[i+1,i] = np.sqrt(i+1)
            
        a = a_dagger.conj().T
        n_hat =np.diag(a_dagger@a)

        B_list = []
        for k in range(self.input_dim):
            prefactor = np.sqrt(1/(factorial(k))*1/g*((g-1)/g)**k) 
            B_list.append(prefactor* np.linalg.matrix_power(a_dagger,k) @ np.diag(g**(-n_hat/2)))
        return B_list

    def make_A(self,eta, n_b, op_idx):
        """
        Constructs the Kraus matrix representing the loss of photons in the system $A_k$
        
        Args:
            eta (float): Transmissivity of the beam splitter coupling to the environment
            n_b (float): Mean thermal photon number.
            state_index (int) : gives the index of the state we create the Kraus matrix for.
            state_dim (int): dimension of the state we should create the Kraus matrix for
        
        Returns:
            list of np.ndarray: returns the Kraus operators applied on the space given by the state index
        """
        g = 1 + (1-eta)*n_b
        tau = eta/g

        print('tau',tau)
        #max photon number
    
        a  = np.zeros((self.input_dim, self.input_dim))
        for i in range(0,self.input_dim-1):
            a[i,i+1] = np.sqrt(i+1)
        a_dagger = a.conj().T
        n_hat = a_dagger@a  

        if op_idx == 0 :
            annihilation_op = np.kron(a,np.eye(self.internal_dim))
            n_hat = np.kron(n_hat, np.eye(self.internal_dim))

        elif op_idx == 1:
            annihilation_op = np.kron(np.eye(self.internal_dim),a)
            n_hat = np.kron( np.eye(self.internal_dim),n_hat)
        elif op_idx == 2:
            annihilation_op = a

      
        A_list = []
        #l is number of lost photons
        for l in range(self.input_dim):
            prefactor = np.sqrt((1-tau)**l/(factorial(l)))
            A_list.append(prefactor * tau**(n_hat/2) * np.linalg.matrix_power(annihilation_op,l))
            
        return A_list 
    
    def make_B(self,eta, n_b, op_idx):
        """
        Constructs the Kraus matrix representing the gain of photons in the system $B_j$
        
        Args:
            eta (float): Transmissivity of the beam splitter coupling to the environment
            n_b (float): Mean thermal photon number.
            op_idx (int): gives the index of the state we create the Kraus matrix for.
            state_dim (int): dimension of the state we should create the Kraus matrix for


        Returns:
            list of np.ndarray: returns the Kraus operators applied on the space given by the state index
        """
        g = 1 + (1-eta)*n_b
        
        a_dagger  = np.zeros((self.input_dim, self.input_dim ))
        for i in range(self.input_dim-1):
            a_dagger[i+1,i] = np.sqrt(i+1)
            
        a = a_dagger.conj().T
        n_hat = a_dagger@a

        if op_idx == 0 :
            creation_op = np.kron(a_dagger,np.identity(self.internal_dim))
            n_hat = np.kron(n_hat, np.identity(self.internal_dim))
        elif op_idx == 1:
            creation_op = np.kron(np.identity(self.internal_dim),a_dagger)
            n_hat = np.kron( np.identity(self.internal_dim),n_hat)

        elif op_idx == 2:
            creation_op = a_dagger
            

        B_list = []
        for k in range(self.input_dim):
            prefactor = np.sqrt(1/(factorial(k))*1/g*((g-1)/g)**k) 
            B_list.append(prefactor* np.linalg.matrix_power(creation_op,k) * g**(-n_hat/2))
    
        return B_list

    def get_AB_kraus(self,eta,nB,op_idx):

        A = self.make_A(eta,nB,op_idx)
        B = self.make_B(eta,nB,op_idx)

        K_list = []
        for i in range(len(A)):
            for j in range(len(B)):
                K_list.append(B[j]@A[i])

        return K_list
    
    def get_AB_kraus_joint(self,eta,nB):
        A = self.make_A_joint(eta,nB)
        B = self.make_B_joint(eta,nB)

        K_list = []
        for i in range(len(A)):
            for j in range(len(B)):
                K_list.append(B[j]@A[i])

        return K_list

    
    def storage_experiment(self):
        kraus = []
        bs = self.make_beamsplitter_U(np.sqrt(self.t_in),0)
        kraus.append(bs)
        return np.array(kraus)
    
    def storage_noise(self):
        kraus = []
        AB = self.get_AB_kraus(self.eta_e,self.nB_e,2)
        kraus.append(AB)
        return np.array(AB)      
        
    def retrieval_experiment(self):
        kraus = []
        bs = self.make_beamsplitter_U(np.sqrt(self.t_out),1)
        kraus.append(bs)
        return np.array(kraus)
    
    def retreival_noise(self):
        kraus = []
        AB = self.get_AB_kraus(self.eta_l,self.nB_l,1)
        kraus.append(AB)
        return np.array(AB)      
        
    def storage_experiment_combined(self):
        kraus = []
        bs1 = self.make_beamsplitter_U(np.sqrt(self.t_in),0)
        AB_e = self.get_AB_kraus_joint(self.eta_e,self.nB_e)

        kraus.append([bs1])
        kraus.append(AB_e)
        return kraus

    def retreival_experiment_combined(self):
        kraus = []

        bs2 = self.make_beamsplitter_U(np.sqrt(self.t_out),1)
        AB_l = self.get_AB_kraus_joint(self.eta_l,self.nB_l)

        kraus.append([bs2])
        kraus.append(AB_l)
        return kraus
    
    def calculate_internal_efficiency(self,storage_time):
        '''We model the decay of the spin wave as an exponential decay on the internal efficiency'''
        return self.int_eff_0*np.exp(-storage_time/self.lifetime)
    
    def calculate_overlap(self):
        '''Here we want to calculate the overlap between the frequency envelopes of the memory and the wavefunction'''
        pass


class Test_Memory(Memory):
    def __init__(self,input_dim, internal_dim):

        super(Test_Memory,self).__init__( 0,0,0, input_dim, internal_dim )
        self.t_in = 0.5
        self.t_out = 0.5  
        self.eta_e  = 0.5
        self.eta_l  = 0.5 
        self.nB_e   = 1
        self.nB_l   = 1
        self.internal_eff = 0.33

        self.wavelength = 895
        self.bandwidth = 1e9
        self.lifetime = 3e-6
        self.polarization = ["V","H"]
        self.retrigger = 1e-6


class Lambda895(Memory):
    #LambdaCsD1
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.33                #From literature
        self.int_e2e_0 = 0.13                #From literature
        self.lifetime = 0.89e-6                  #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)
        # nB   = float(nb_dict['LambdaCsD1'])                   #We fit these values in the jupyter notebook.
        self.mu1 = 0.07
        self.nB = self.mu1*self.int_eff_0/(1-transmission)
        self.SNR = 1/self.mu1
        #self.nB = self.

        #self.target_noise_photons = 1/self.SNR
        #self.fit_range  = [0.01,0.1]


        self.wavelength = 895                 #nm
        self.bandwidth = 1e9                  #Hz

        self.polarization = ["V","H"]
        self.retrigger = 11e-6

        self.ref = 5
        self.ref_full = 'Optimization and readout-noise analysis of a warm-vapor electromagnetically-induced-transparency memory on the Cs D1 line \
                        Luisa Esguerra, Leon Messner, Elizabeth Robertson, Norman Vincenz Ewald, Mustafa Guendogan, and Janik Wolters \
                        Phys. Rev. A 107, 042607 (2023) - Published 12 April, 2023'

        super(Lambda895,self).__init__(int_eff,transmission, self.nB, input_dim, internal_dim)


class Ladder780(Memory):
    #FLAME2_Rb
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.51                #From literature
        self.int_e2e_0 = 0.35                #From literature
        self.lifetime = 108e-9                 #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)

        #nB   = float(nb_dict['FLAME2_Rb'])                     #We fit these values in the jupyter notebook.
        #self.SNR = 3300
        #self.target_noise_photons = 1/self.SNR
        #self.fit_range  = [1e-5,1e-4]

        self.mu1 = 3e-6
        self.nB = self.mu1*self.int_eff_0/(1-transmission)
        self.SNR = 1/self.mu1


        self.wavelength = 780                 #nm
        self.bandwidth = 370e6                #Hz
        
        self.polarization = ['L','R']
        self.retrigger = 108e-9               #s


        self.ref = 3
        self.ref_full = 'Davidson, O., Yogev, O., Poem, E. et al. Fast, noise-free atomic optical memory with 35-percent end-to-end efficiency. Commun Phys 6, 131 (2023)'

        super(Ladder780,self).__init__(int_eff,transmission, self.nB, input_dim, internal_dim)

class Ladder852(Memory):
    #ORCA_Cs_D2
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.17                #From literature
        self.int_e2e_0 = 0.049                #From literature
        self.lifetime = 5.4e-9                 #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)
        #nB   = float(nb_dict['ORCA_Cs_D2'])                  #We fit these values in the jupyter notebook.
        #self.SNR = 2900
        #self.target_noise_photons = 1/self.SNR
        #self.fit_range  = [1e-4,1e-3]

        self.mu1 = 3.8e-5
        self.nB = self.mu1*self.int_eff_0/(1-transmission)
        self.SNR = 1/self.mu1

        self.wavelength = 852                #nm
        self.bandwidth = 1e9                #Hz

        self.polarization = ['H','V']
        self.retrigger = 12.5e-9            #s
 
        self.ref = 11
        self.ref_full = 'Kaczmarkek et al. High-speed noise-free optical quantum memory. Phys. Rev. A 97 (2018).'

        super(Ladder852,self).__init__(int_eff,transmission, self.nB, input_dim, internal_dim)
        
class Lambda795(Memory):
    #Lambda_Rb_D1
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.047               #From literature
        self.int_e2e_0 = 0.014                #From literature
        self.lifetime = 680e-9                 #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)
        #nB   = float(nb_dict['Lambda_Rb_D1'])                 #We fit these values in the jupyter notebook.
        #self.SNR = 10.8
        #self.target_noise_photons = 1/self.SNR
        #self.fit_range  = [0.005,0.05]

        self.mu1 = 2.4e-5
        self.nB = self.mu1*self.int_eff_0/(1-transmission)
        self.SNR = 1/self.mu1


        self.wavelength = 795                #nm
        self.bandwidth = 370e6               #Hz

        self.polarization = ['L','R']
        self.retrigger = 2.7e-6              #s

        self.ref = 6
        self.ref_full = 'Buser et al. Single-Photon Storage in a Ground-State Vapor Cell Quantum Memory. PRX Quantum 3, 020349 (2022)'

        super(Lambda795,self).__init__(int_eff,transmission, self.nB, input_dim, internal_dim)

class Ladder895(Memory):
    #FLAME_Cs_D1
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.21                #From literature
        self.int_e2e_0 = 0.027                #From literature
        self.lifetime = 32e-9                 #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)
        #nB   = float(nb_dict['FLAME_Cs_D1'])                 #We fit these values in the jupyter notebook.
        #self.SNR = 830
        #self.target_noise_photons = 1/self.SNR
        #self.fit_range  = [1e-3,1e-2]

        self.mu1 = 7.2e-5 
        self.nB = self.mu1*self.int_eff_0/(1-transmission)
        self.SNR = 1/self.mu1

        self.wavelength = 895                # nm
        self.bandwidth = 560e6               # Hz

        self.polarization =  ['H','V']
        self.retrigger = 33e-9               # s

        self.ref = 2
        self.ref_full = 'Maass et al. Room-temperature ladder-type optical memory compatible with single photons from semiconductor quantum dots. Phys. Rev. Applied 22, 044050 (2024)'

        super(Ladder895,self).__init__(int_eff,transmission, self.nB, input_dim, internal_dim)


class LambdaCsD2Cold(Memory):
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.70                #From literature
        self.int_e2e_0 =  self.int_eff_0 * 0.5               #not given in literature, estimate assuming eta_{trans} = 0.5
        self.lifetime = 14e-6                 #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)
        nB   = None                 #We fit these values in the jupyter notebook.

        self.SNR = None
        self.target_noise_photons = 1/self.SNR
        self.fit_range  = [0.005,0.03]


        self.memory_envelope = 'gaussian'

        self.wavelength = 852                #nm
        self.bandwidth = 2e6                #Hz

        self.polarization = ['L','R']
        self.retrigger = 50e-3              #


        self.ref = 5
        self.ref_full = 'Vernaz-Gris, P., Huang, K., Cao, M. et al. Highly-efficient quantum memory for polarization qubits in a spatially-multiplexed cold atomic ensemble. Nat Commun 9, 363 (2018).'

        super(LambdaCsD2Cold,self).__init__(int_eff,transmission, nB, input_dim, internal_dim)


class Lambda795Compact(Memory):
    #LambdaRbD1Compact
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.25               #From literature
        self.int_e2e_0 =  self.int_eff_0 * 0.5               #not given in literature, estimate assuming eta_{trans} = 0.5
        self.lifetime = 180e-6                 #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)
        #nB   = float(nb_dict['LambdaRbD1Compact'])                 #We fit these values in the jupyter notebook.
        #self.SNR = 17
        #self.target_noise_photons = 1/self.SNR
        #self.fit_range  = [1e-2,1e-1]

        self.mu1 =1.9e-3
        self.nB = self.mu1*self.int_eff_0/(1-transmission)
        self.SNR = 1/self.mu1


        self.wavelength = 795                #nm
        self.bandwidth = 2e6                #Hz

        self.polarization = ['L','R']
        self.retrigger = 5e-3

        self.ref = 8
        self.ref_full = 'Wang et al. Field-Deployable Quantum Memory for Quantum Networking. Phys. Rev. Applied 18 (2022)'

        super(Lambda795Compact,self).__init__(int_eff,transmission, self.nB, input_dim, internal_dim)

class Lambda895Compact(Memory):
    #LambdaCsD1Compact
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.23                #From literature
        self.int_e2e_0 = 0.054               #From literature
        self.lifetime = 2.4e-6                 #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)
        #nB   = float(nb_dict['LambdaCsD1Compact'])                 #We fit these values in the jupyter notebook.
        #self.SNR = 14
        #self.target_noise_photons = 1/self.SNR
        #self.fit_range  = [1e-2,1e-1]

        self.mu1 = 0.06
        self.nB = self.mu1*self.int_eff_0/(1-transmission)
        self.SNR = 1/self.mu1


        self.wavelength = 895                #nm
        self.bandwidth = 44e6                #Hz

        self.polarization =  ['H','V']
        self.retrigger = 32.7e-6

        self.ref = 1
        self.ref_full = 'Jutisz et al. Stand-alone mobile quantum memory system. Phys. Rev. Applied 23 (2025)'

        super(Lambda895Compact,self).__init__(int_eff,transmission, self.nB, input_dim, internal_dim)

class Ladder1529(Memory):
    #ORCA_Telecom
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.21                #From literature
        self.int_e2e_0 = 0.094               #From literature
        self.lifetime = 1.1e-9               #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)
       
        #nB   = float(nb_dict['ORCA_Telecom'])                 #We fit these values in the jupyter notebook.
        #self.SNR = 19000
        #self.target_noise_photons = 1/self.SNR
        #self.fit_range  = [5e-6,5e-5]


        self.mu1 = 4.4e-6 
        self.nB = self.mu1*self.int_eff_0/(1-transmission)
        self.SNR = 1/self.mu1


        self.wavelength = 1529               #nm
        self.bandwidth = 1e9                #Hz

        self.polarization =  ['H','V']
        self.retrigger = 12.5e-9


        self.ref = 4
        self.ref_full = 'Thomas et al.Single-Photon-Compatible Telecommunications-Band Quantum Memory in a Hot Atomic Gas. Phys. Rev. Applied 19, L031005  (2023)'

        super(Ladder1529,self).__init__(int_eff,transmission, self.nB, input_dim, internal_dim)


class Lambda780Superradiance(Memory):
    #SuperradianceRbD2
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.03                #From literature
        self.int_e2e_0 = self.int_eff_0 * 0.5               #not given in literature, estimate assuming eta_{trans} = 0.5
        self.lifetime = 4.7e-9               #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)
       
        # nB   = float(nb_dict['SuperradianceRbD2'])                 #We fit these values in the jupyter notebook.
        # self.SNR = 40
        # self.target_noise_photons = 1/self.SNR
        # self.fit_range  = [1e-4,1e-3]


        self.mu1 =2.1e-4
        self.nB = self.mu1*self.int_eff_0/(1-transmission)
        self.SNR = 1/self.mu1


        self.wavelength = 780                #nm
        self.bandwidth = 12.7e6                #Hz

        self.polarization =  ['H','V']
        self.retrigger = 5.7e-6

        self.ref = 7
        self.ref_full = 'Rastogi et al. Superradiance-Mediated Photon Storage for Broadband Quantum Memory. Phys. Rev. Lett. 129, 120502 (2022)'
 
        super(Lambda780Superradiance,self).__init__(int_eff,transmission, self.nB, input_dim, internal_dim)



class SpinExchangeCsD1(Memory):
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.09                #From literature
        self.int_e2e_0 = None               #From literature
        self.lifetime = 149e-3               #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)
        nB   = None                 #We fit these values in the jupyter notebook.

        self.SNR = None
        self.target_noise_photons = 1/self.SNR
        self.fit_range  = [1e-5,1e-4]


        self.memory_envelope = 'gaussian'

        self.wavelength = 894                #nm
        self.bandwidth = None                #Hz

        self.polarization =  ['H','V']


        self.ref = 10
        self.ref_full = 'Katz, O., Firstenberg, O. Light storage for one second in room-temperature alkali vapor. Nat Commun 9, 2074 (2018). '
 
        super(SpinExchangeCsD1,self).__init__(int_eff,transmission, nB, input_dim, internal_dim)
        
class Lambda780RydbergSource(Memory):
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.21               #From literature
        self.int_e2e_0 =  self.int_eff_0 * 0.5               #not given in literature, estimate assuming eta_{trans} = 0.5
        self.lifetime = 1.2e-6                 #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)
        
        self.mu1 = 1e-3
        self.nB = self.mu1*self.int_eff_0/(1-transmission)
        self.SNR = 1/self.mu1

        self.wavelength = 780                #nm
        self.bandwidth = 17.6e6                #Hz

        self.polarization = ['L','R']
        self.retrigger = 11e-3              #s

        self.ref = 9
        self.ref_full = 'Heller, L.,  Lowinski, J., Theophilo, K. et al. Raman Storage of Quasideterministic Single Photons Generated by Rydberg Collective Excitations in a Low-Noise Quantum Memory, Phys. Rev. Applied 18 (2022).'


        super(Lambda780RydbergSource,self).__init__(int_eff,transmission, self.nB, input_dim, internal_dim)

class Lambda780BEC(Memory):
    def __init__(self,input_dim, internal_dim, storage_duration):
        self.int_eff_0 = 0.3                #From literature
        self.int_e2e_0 =  self.int_eff_0 * 0.5               #not given in literature, estimate assuming eta_{trans} = 0.5
        self.lifetime = 15.8e-6                 #s

        transmission = self.int_e2e_0/self.int_eff_0

        int_eff = self.calculate_internal_efficiency(storage_duration)
        
        self.mu1 = 5e-3
        self.nB = self.mu1*self.int_eff_0/(1-transmission)
        self.SNR = 1/self.mu1

        self.wavelength = 780               #nm
        self.bandwidth = 22e6                #Hz

        self.polarization = ['L','R']
        self.retrigger = 20              #s


        self.ref = 10
        self.ref_full = 'Saglanyurek, P., Huang, K., Cao, M. et al. Highly-efficient quantum memory for polarization qubits in a spatially-multiplexed cold atomic ensemble. Nat Commun 9, 363 (2018).'

        super(Lambda780BEC,self).__init__(int_eff,transmission, self.nB, input_dim, internal_dim)

        
#cs = LambdaCsD1( 2, 2, 0.5e-6)
#kraus = cs.storage_experiment()
