from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
import time

# MATERIALS and their refractive index functions.
# This will be used to get refractive indices of the materials.
number_of_data = 20000 # Change this, and it will create this number of data.
# It will automatically save the files. You just need to run READ_FILES.py file.

class MATERIALS:

    def __init__(self, lamda):
        self.lamda = lamda


    def Ag(self):
        n2 = 1.78522 + ((1.21202*self.lamda**2)/(self.lamda**2 - 0.01262)) - (0.01681 * self.lamda**2)
        n = np.sqrt(n2)
        return n


    def BK7(self):
        term1 = 1
        term2 = ((1.03961212 * self.lamda ** 2) / (self.lamda ** 2 - 0.00600069867))
        term3 = ((0.231792344 * self.lamda ** 2) / (self.lamda ** 2 - 0.0200179144))
        term4 = ((1.01046945 * self.lamda ** 2) / (self.lamda ** 2 - 103.560653))
        result = term1 + term2 + term3 + term4
        n = np.sqrt(result)
        return n


    def BAF10(self):
        term1 = 1
        term2 = 1.5851495 * self.lamda ** 2 / (self.lamda ** 2 - 0.00926681282)
        term3 = 0.143559385 * self.lamda ** 2 / (self.lamda ** 2 - 0.0424489805)
        term4 = 1.08521269 * self.lamda ** 2 / (self.lamda ** 2 - 105.613573)
        result = term1 + term2 + term3 + term4
        n = np.sqrt(result)
        return n


    def BAK1(self):
        term1 = 1
        term2 = 1.12365662 * self.lamda ** 2 / (self.lamda ** 2 - 0.00644742752)
        term3 = 0.309276848 * self.lamda ** 2 / (self.lamda ** 2 - 0.0222284402)
        term4 = 0.881511957 * self.lamda ** 2 / (self.lamda ** 2 - 107.297751)
        result = term1 + term2 + term3 + term4
        n = np.sqrt(result)
        return n


    def FK51A(self):
        term1 = 1
        term2 = 0.971247817 * self.lamda ** 2 / (self.lamda ** 2 - 0.00472301995)
        term3 = 0.216901417 * self.lamda ** 2 / (self.lamda ** 2 - 0.0153575612)
        term4 = 0.904651666 * self.lamda ** 2 / (self.lamda ** 2 - 168.68133)
        result = term1 + term2 + term3 + term4
        n = np.sqrt(result)
        return n


    def LASF9(self):
        term1 = 1
        term2 = 2.00029547 * self.lamda**2 / (self.lamda**2 - 0.0121426017)
        term3 = 0.298926886 * self.lamda**2 / (self.lamda**2 - 0.0538736236)
        term4 = 1.80691843 * self.lamda**2 / (self.lamda**2 - 156.530829)
        result = term1 + term2 + term3 + term4
        n = np.sqrt(result)
        return n


    def SF5(self):
        term1 = 1
        term2 = 1.52481889 * self.lamda ** 2 / (self.lamda ** 2 - 0.011254756)
        term3 = 0.187085527 * self.lamda ** 2 / (self.lamda ** 2 - 0.0588995392)
        term4 = 1.42729015 * self.lamda ** 2 / (self.lamda ** 2 - 129.141675)
        result = term1 + term2 + term3 + term4
        n = np.sqrt(result)
        return n


    def SF10(self):
        term1 = 1
        term2 = 1.62153902 * self.lamda ** 2 / (self.lamda ** 2 - 0.0122241457)
        term3 = 0.256287842 * self.lamda ** 2 / (self.lamda ** 2 - 0.0595736775)
        term4 = 1.64447552 * self.lamda ** 2 / (self.lamda ** 2 - 147.468793)
        result = term1 + term2 + term3 + term4
        n = np.sqrt(result)
        return n


    def SF11(self):
        term1 = 1
        term2 = 1.73759695 * self.lamda ** 2 / (self.lamda ** 2 - 0.013188707)
        term3 = 0.313747346 * self.lamda** 2 / (self.lamda ** 2 - 0.0623068142)
        term4 = 1.89878101 * self.lamda ** 2 / (self.lamda ** 2 - 155.23629)
        result = term1 + term2 + term3 + term4
        n = np.sqrt(result)
        return n


    def FusedSilica(self):
        term1 = 1
        term2 = 0.6961663 * self.lamda ** 2 / (self.lamda ** 2 - 0.0684043 ** 2)
        term3 = 0.4079426 * self.lamda ** 2 / (self.lamda ** 2 - 0.1162414 ** 2)
        term4 = 0.8974794 * self.lamda ** 2 / (self.lamda ** 2 - 9.896161 ** 2)
        result = term1 + term2 + term3 + term4
        n = np.sqrt(result)
        return n


    def SCHOTTMP(self):
        n = np.sqrt(1 + (1.3182408*self.lamda**2) / (self.lamda**2 - 0.00879) - (0.0244*self.lamda**2) / (self.lamda**2 - 0.0609) - (1.08915181*self.lamda**2) / (self.lamda**2 - 110))
        return n


    def SCHOTTK7(self):
        n = np.sqrt(1 + (1.1273555*self.lamda**2) / (self.lamda**2 - 0.00720341707) - (0.124412303*self.lamda**2) / (self.lamda**2 - 0.0269835916) - (0.827100531*self.lamda**2) / (self.lamda**2 - 100.384588))
        return n


    def SCHOTTSK(self):
        n = np.sqrt(1 + (1.28189012 * self.lamda ** 2) / (self.lamda ** 2 - 0.0072719164) - (0.257738258 * self.lamda ** 2) / (
                    self.lamda ** 2 - 0.0242823527) - (0.96818604 * self.lamda ** 2) / (self.lamda ** 2 - 110.377773))
        return n


    def BeAl2O4(self):
        n = np.sqrt(1.78522 + (1.21202*self.lamda**2) / (self.lamda**2 - 0.01262) - 0.01681*self.lamda**2)
        return n


    def BeAl6O10(self):
        n = np.sqrt((2.986556 + 0.01828907 / self.lamda ** 2 - 0.01445419 * self.lamda ** 2))
        return n


    def CaGdAlO4(self):
        n = np.sqrt(3.64863 + 0.03926 / (self.lamda**2 - 0.02239) - 0.01121 * self.lamda**2)
        return n


    def MgAl2O4(self):
        n = np.sqrt(1 + (1.8938*self.lamda**2) / (self.lamda**2 - 0.09942**2) - (3.0755*self.lamda**2) / (self.lamda**2 - 15.826**2))
        return n


    def CdGeAs2(self):
        n = np.sqrt(1 + 9.1064 - (2.2988*self.lamda**2) / (self.lamda**2 - 1.0872) - (1.6247*self.lamda**2) / (self.lamda**2 - 1370))
        return n

    #def ZnSiAs2 (self):
        #n = np.sqrt(1 + 3.6006 + ((5.6912*self.lamda**2) / (self.lamda**2 - 0.1437)) - (1.1316*self.lamda**2) / (self.lamda**2 - 700))
        #return n

    def BaB2O4(self):
        n = np.sqrt(2.7405 + 0.0184 / (self.lamda**2 - 0.0179) - 0.0155 * self.lamda**2)
        return n


    def BaF2(self):
        n = np.sqrt(1 + 0.33973 - (0.81070 * self.lamda**2) / (self.lamda**2 - 0.10065**2) -
                    (0.19652 * self.lamda**2) / (self.lamda**2 - 29.87**2) - (4.52469 * self.lamda**2) /
                    (self.lamda**2 - 53.82**2))
        return n


    def Al2O3(self):
        n = np.sqrt(1 + (1.4313493 * self.lamda**2) / (self.lamda**2 - 0.0726631**2) - (0.65054713 * self.lamda**2) / (self.lamda**2 - 0.1193242**2) - (5.3414021 * self.lamda**2) / (self.lamda**2 - 18.028251**2))
        return n


def get_info(material_names, bottom=0.3, upper=0.9, datapoints=100):
    info_dict = {}

    # Generate an array of wavelengths
    wavelengths = np.linspace(bottom, upper, datapoints)

    # Iterate over each material name
    for material_name in material_names:
        # Instantiate the MATERIALS class with the wavelength array
        material_instance = MATERIALS(wavelengths)

        # Get the function corresponding to the material name
        material_function = getattr(material_instance, material_name)

        # Calculate refractive indices for the material
        refractive_indices = material_function()

        # Store wavelengths and refractive indices in a dictionary
        info_dict[material_name] =(wavelengths,refractive_indices)

    return info_dict

def create_matrix(materials, thickness, upper, bottom, data_points):
    """
    Creates a matrix for reflectance calculation.

    Args:
        materials (list): List of materials.
        thickness (list): List of thickness values.
        upper (float): Upper limit for wavelength.
        bottom (float): Bottom limit for wavelength.
        data_points (int): Number of data points.

    Returns:
        list: List of reflectance values.
    """
    lamda = np.linspace(bottom, upper, data_points)  # Create wavelength data with the increment of 0.01 nm
    gammai = get_info(materials, upper=upper, bottom=bottom,
                      datapoints=data_points)  # it will take refractive index value of each material

    gamma_0 = 1.0  # basicly n0 = 1
    gamma_s = 1.52  # ns = 1.52     # Later, these values will be taken from user.

    R = []  # List for reflectance values (between 0 and 1)
    for wavelentgh in range(len(lamda)):
        MT = np.identity(2)  # Initiate with identity matrix.
        for material in range(len(materials)):
            phase = (2 * np.pi / float(lamda[wavelentgh])) * float(
                gammai[materials[material]][1][wavelentgh]) * float(thickness[material])

            M1 = np.array(
                [[np.cos(phase), complex(0, np.sin(phase)) / float(gammai[materials[material]][1][wavelentgh])],
                 [float(gammai[materials[material]][1][wavelentgh]) * complex(0, np.sin(phase)), np.cos(phase)]],
                dtype=complex)

            MT = np.dot(MT, M1)  # Calculate MT = M1*M2*M3 ... Mn

        ri = (((float(gamma_0) * MT[0, 0]) + (float(gamma_0) * float(gamma_s) * MT[0, 1]) - (MT[1, 0]) - (
                    float(gamma_s) * MT[1, 1])) /
              ((float(gamma_0) * MT[0, 0]) + (float(gamma_0) * float(gamma_s) * MT[0, 1]) + (MT[1, 0]) + (
                          float(gamma_s) * MT[1, 1])))
        # Calculate r using the formula 19-36.
        R.append((np.dot(ri, ri.conjugate()).real))

    return R


def generate_data(seed_num):
    random.seed(seed_num)
    Material_names = [func for func in dir(MATERIALS) if callable(getattr(MATERIALS, func)) and not func.startswith("__")]

    # Ensure at least two materials are chosen
    num_materials = max(2, 12)#len(Material_names))

    # Shuffle the material names to ensure randomness
    random.shuffle(Material_names)

    # Take a random number of material names (up to the total number of materials)
    random_material_names = Material_names[:random.randint(2,num_materials)]

    # Create a list of random lengths with the same size as random_material_names
    random_lengths = [random.randint(1, 100) / 100 for _ in range(len(random_material_names))]

    return np.array(random_material_names), np.array(random_lengths)

def encode_methods(methods):
    """
    Encode a list of methods using one-hot encoding.

    Parameters:
    methods (list): A list of method names.

    Returns:
    numpy.ndarray: One-hot encoded array of the methods.
    """
    # Reshape the methods list to a numpy array
    methods_array = np.array(methods).reshape(-1, 1)

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder()

    # Fit and transform the data
    onehot_encoded = encoder.fit_transform(methods_array)

    # Convert the encoded data to an array
    onehot_encoded_methods = onehot_encoded.toarray()

    return onehot_encoded_methods

def Name_to_one_hot(array):
  # Example usage:
  METHODS = np.array(['Ag', 'Al2O3', 'BAF10', 'BAK1', 'BK7', 'BaB2O4', 'BaF2', 'BeAl2O4', 'BeAl6O10', 'CaGdAlO4', 'CdGeAs2', 'FK51A', 'FusedSilica', 'LASF9', 'MgAl2O4', 'SCHOTTK7', 'SCHOTTMP', 'SCHOTTSK', 'SF10', 'SF11', 'SF5'])
  encoded_methods = encode_methods(METHODS)
  indices = [np.where(METHODS == element)[0][0] for element in array]

  one_hot_encoded_materials = encode_methods(METHODS)
  array_list = []

  return one_hot_encoded_materials[indices]


def Random_R_Generator(bottom,upper,datapoints,number_of_data):
    REF = []
    MATERIALS_GENERATED = []
    THICKNESSES_GENERATED = []
    WAVELENGTH = np.linspace(bottom,upper,datapoints)
    for i in range(number_of_data):
        # Generate random seed
        r = random.randint(1, 123)

        # Generate data
        Materials, thickness = generate_data(r)
        noise_range = [-0.02,0.02]
        # Create matrix
        R = create_matrix(materials=Materials, thickness=thickness, upper=upper, bottom=bottom, data_points=datapoints)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=np.asarray(R).shape)
        R_with_noise = R + noise

        #x=l[20:]; y=R[20:]
        #RR = [round(i,5) for i in R] # If you need to decrease the size of the Refraction_values.txt, You can uncomment this part.
        REF.append(R_with_noise)
        one_hot_version_of_materials = Name_to_one_hot(Materials)
        MATERIALS_GENERATED.append(one_hot_version_of_materials)
        THICKNESSES_GENERATED.append(thickness)

    REF = np.array(REF)

    return REF, MATERIALS_GENERATED, THICKNESSES_GENERATED, WAVELENGTH


start_time = time.time()
print(0)
Refraction , materials, thicks , wavelength_micro = Random_R_Generator(0.280,0.900,200,number_of_data)
end_time = time.time()
print(end_time - start_time)

start_time = time.time()
print(0)
def write_lists_to_file(filename, *lists):
    with open(filename, 'w') as file:
        for row in zip(*lists):
            file.write('\t'.join(map(str, row)) + '\n')

# Example usage
write_lists_to_file(f"Refraction_values_{int(number_of_data/1000)}k.txt",Refraction)
write_lists_to_file(f"Materials_list_{int(number_of_data/1000)}k.txt",materials)
write_lists_to_file(f"Thickness_values_{int(number_of_data/1000)}k.txt",thicks)
end_time = time.time()
print(end_time - start_time)



