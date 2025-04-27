import torch # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import torch.nn as nn # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def get_file(file_name):
    dp = np.loadtxt(file_name,delimiter=',') 
  
    ###### Set data to 32 bits to save memory
    data = dp.astype(np.float32)
    pd.set_option('display.precision', 5)

    ###### Convert to DataFrame
    data = pd.DataFrame(data)
  
    ###### Kepp MHz, Azimuth, and Elevation
    first_three = data.iloc[:, 0:3]

    ###### Convert real and imaginary pairs to complex
    complex_columns = []
    for i in range(3, 19, 2):  
        real = data.iloc[:, i]
        imag = data.iloc[:, i + 1]
        complex_col = real.values + 1j * imag.values
        complex_columns.append(complex_col)
  

    ###### Stack into a DataFrame
    complex_df = pd.DataFrame(np.column_stack(complex_columns))
    
    ###### Concatenate the columns together
    divide_data = len(data) // 1  # Reduce data to 1/n amount
    
    data = pd.concat([first_three, complex_df], axis=1)
    data = data.iloc[:divide_data]  # Keep only the first half of the rows
    
    channel_values = data.iloc[:, 3:]
    true_values = data.iloc[:, 1:3]
    
    
    ###### Get data and transform it to complex input data for torch
    X_train = torch.from_numpy(channel_values.to_numpy()).to(torch.cfloat)
    
    ###### Convert true values (Azimuth and Elevation) to unit vectors
    az_rad = torch.deg2rad(torch.from_numpy(true_values.iloc[:, 0].to_numpy()).float())
    el_rad = torch.deg2rad(torch.from_numpy(true_values.iloc[:, 1].to_numpy()).float())
    
    ###### Convert Azimuth and Elevation torch variables to true values to be compared and tested from
    az_complex = torch.cos(az_rad) + 1j * torch.sin(az_rad)
    el_complex = torch.cos(el_rad) + 1j * torch.sin(el_rad)
    
    ###### Now create variable that holds targeted values
    Y_train = torch.stack([az_complex, el_complex], dim=1)  
    


    ###### Normalize inputs, with zero mean and unit variance, according to articles description
    if X_train.numel() > 0:
    
        ### Compute mean for each columns
        mean_x = X_train.mean(dim=0)
        
        ### Subtract mean to x features to center data
        X_train = X_train - mean_x 
        
        ### Compute standard deviation for real and imaginary
        std_x_real = X_train.real.std(dim=0)
        std_x_imag = X_train.imag.std(dim=0)
        
        ### Avoid division by 0, so add small std value
        eps = 1e-8
        
        ### Combine real and imag std and divide features to unit standard
        std_x = torch.sqrt(np.square(std_x_real) + np.square(std_x_imag) + eps)
        X_train = X_train / std_x

    
    return X_train, Y_train, true_values
###############################################################################################
################## GET DATA AND GET ITS COLUMNS 2-11 ##################
###### Get data
################## GET DATA ##################
###############################################################################################


###############################################################################################
################## SETP UP ARCHITECTURE FOR C-RBF ##################
class DeepComplexRBF(nn.Module):
    def __init__(self, input_dim, hidden_neurons, output_dim):

        self.t = 1

        ###### Call nn.Module() to intialize parameters for Pytorch
        super(DeepComplexRBF, self).__init__()
      
        ###### Set up counts for layers, neurons, inputs, and outputs
        self.hidden_neurons = hidden_neurons
        self.num_layers = len(self.hidden_neurons)
        self.input_dim = input_dim
        self.output_dim = output_dim

        ###### Initialize I and O, where I is neurons and O is incoming outputs from previous layer to current layer
        O = [None] * (self.num_layers + 1)  
        I = [None] * (self.num_layers + 1)  

        ###### Set input size to O in first layer
        O[0] = input_dim

        ###### Compute size I and O for each layer
        for l in range(1, self.num_layers+1):
            I[l] = self.hidden_neurons[l-1]

            if l < self.num_layers:  
                O[l] = I[l]   
            
            else:
                O[l] = output_dim  

        ###### Register parameters for each layer and assign complex data type to torch
        self.weights = nn.ParameterList()
        self.biases  = nn.ParameterList()
        self.centers = nn.ParameterList()
        self.sigmas  = nn.ParameterList()

        self.weights_m = []
        self.weights_v = []

        self.biases_m = []
        self.biases_v = []

        self.centers_m = []
        self.centers_v = []

        self.sigmas_m = []
        self.sigmas_v = []

        ############ Loop trough each Layer and intialize the parameters for each layer
        for l in range(1, self.num_layers+1):
          
            ############ Initialize size for weights, basis, centers (Gamma), and Variance in each Layer
            # Weights to compute with neurons
            W_shape = (O[l], I[l])
            W_param = nn.Parameter(torch.zeros(W_shape, dtype=torch.cfloat))   

            ###### Bias for each ouput in layer
            b_param = nn.Parameter(torch.zeros(O[l], dtype=torch.cfloat))

            ###### Gama for RBF centers
            center_shape = (I[l], O[l-1])
            center_points = nn.Parameter(torch.zeros(center_shape, dtype=torch.cfloat))

            ###### Variance for each neuron in layer
            sigma_param = nn.Parameter(torch.zeros(I[l], dtype=torch.float32))

            ############ Set temporary values
            ###### Bias intialized to 0 according to article
            b_param.data.fill_(0)  
          
            ###### Variances initialized to 1.0 according to article
            sigma_param.data.fill_(1.0)
          
            ###### For weights, set up some random weight to weights in each layers
            ### Use Xavier-like initialization, found online
            real_imag_std = 1.0 / np.sqrt(I[l])
            W_param.data = (torch.randn(W_shape) * real_imag_std) + 1j * (torch.randn(W_shape) * real_imag_std)

            ####### Creating centers, of which is number of inputs from previous layer, seen in equation 32
            std_c = 1.0 / np.sqrt(2 * O[l - 1])
            center_points.data = (torch.randn(center_shape) * std_c) + 1j * (torch.randn(center_shape) * std_c)

            self.weights.append(W_param)
            self.biases.append(b_param)
            self.centers.append(center_points)
            self.sigmas.append(sigma_param)


            self.weights_m.append(torch.zeros_like(W_param))
            self.weights_v.append(torch.zeros(W_param.shape, dtype=torch.float32, device=W_param.device))

            self.biases_m.append(torch.zeros_like(b_param))
            self.biases_v.append(torch.zeros(b_param.shape, dtype=torch.float32, device=b_param.device))

            self.centers_m.append(torch.zeros_like(center_points))
            self.centers_v.append(torch.zeros(center_points.shape, dtype=torch.float32, device=center_points.device))

            self.sigmas_m.append(torch.zeros_like(sigma_param))
            self.sigmas_v.append(torch.zeros(sigma_param.shape, dtype=torch.float32, device=sigma_param.device))


        """
        for i in range(1, len(self.weights_m)+1):
            print("Layer: ", i)
            print("Weights: ", self.weights_m[i-1].shape)
            print("Biase", self.biases_m[i-1].shape)
            print("Centers",self.centers_m[i-1].shape)
            print("Sigmas", self.sigmas_m[i-1].shape)
            print(self.weights[i-1].shape)
            print(self.biases[i-1].shape)
            print(self.centers[i-1].shape)
            print(self.sigmas[i-1].shape)
            print()
        """



        
################## SETP UP ARCHITECTURE FOR C-RBF ##################
###############################################################################################


###############################################################################################
################## TRAIN AND SET PARAMETERS ##################
    def training_step(self, x, target, lr, t):
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = t
        
        ############ Intialize variables to compute forward propagation
        ###### Convert data to Complex Values in torch to be computed on
        x = x.to(torch.cfloat)
        target = target.to(torch.cfloat)

        ###### Intialize storage variable "y" to hold Output and "phi" RBF activations in each layer
        y = [None] * (self.num_layers + 1)   
        phi = [None] * (self.num_layers + 1) 
        
        ###### Store first x value to y, will be used for y[l-1] layer
        y[0] = x

        ############ Forward loop and compute parameters for each layer
        for l in range(1, self.num_layers+1):

            ###### Retrieve paramaters
            W = self.weights[l-1]
            b = self.biases[l-1]
            Gamma = self.centers[l-1]
            sigma = self.sigmas[l-1]

            ###### Compute diff, turn y^l-1 shape from (O[l-1]) to (1, O[l-1])
            diff = y[l-1].unsqueeze(-2) - Gamma 
          
            ###### Compute Square Euclidean distance, where we only add columns,  (I[l],)
            dist_sq = (diff.real**2 + diff.imag**2).sum(dim=-1)  

            ###### Compute phi, becoming real since ||a+jb||^2 = (a^2) + (b^2) with shape 
            phi[l] = torch.exp(- dist_sq / (2.0 * sigma))  
            
            ###### Compute y^l = W * phi + b  
            phi_l_complex = phi[l].to(torch.cfloat)
            y[l] = W @ phi_l_complex + b  
        





        ############ Intialize variables for backpropagation
        ###### Compute error between target and results
        output = y[self.num_layers]
        error = target - output  

        ###### Assign learning rate to several variables
        lr_w = lr_b = lr_gamma = lr_sigma = lr
      
        ###### Initialize variables to help update Weights, Biases, Gamma, and Sigma
        psi = [None] * (self.num_layers + 1)
        delta = [None] * (self.num_layers + 1)
        
        ###### Get num of layers and set error to psi
        L = self.num_layers
        psi[L] = error  

        ############ Get data to compute backpropagation
        ###### Backpropagate through layers L to 1
        for l in range(L, 0, -1):

            ############### Get parameters and initalize var. to compute backpropagation 
            ###### Getting gradient decent for central and variance by getting "xi" first according to article
            W = self.weights[l-1]
            psi_l = psi[l]
          
            ###### Computing xi
            xi = (W.real.t() @ psi_l.real) + (W.imag.t() @ psi_l.imag) 
          
            ###### Compute beta
            phi_l = phi[l]           
            sigma_l = self.sigmas[l-1]  
            beta = phi_l / sigma_l    
          
            ###### Compute delta
            delta[l] = - xi * beta   
            
            ###### Compute errors for previous layer, the psi[l-1], ignore last layer when we reach there
            if l > 1:

                ### Collect gamma, the center of current layer
                Gamma_l = self.centers[l-1]  
                ### Expand y to match to Gamma and subtratc, shape (I[l], O[l-1])
                Y_expand = y[l-1].expand_as(Gamma_l)
                diff_l = Y_expand - Gamma_l  
                ### Compute Backpropagate for errors in previous layer, shape (O[l-1],) 
                psi[l-1] = diff_l.T @ delta[l].to(torch.cfloat)

            ############### Compute backpropagation
            """
            M = self.m_momentum[l-1]
            V = self.v_momentum[l-1]
            """
            ###### Update weights, shape (O[l], I[l])



            dW = psi_l.unsqueeze(1) * phi_l.unsqueeze(0)

            #print(W.data)
            self.weights_m[l-1] = beta1 * self.weights_m[l-1] + (1-beta1) * dW
            self.weights_v[l-1] = (beta2 * self.weights_v[l-1] + (1 - beta2) * (dW.abs() ** 2))


            m_W_hat = self.weights_m[l-1] / (1 - beta1 ** t)
            v_W_hat = self.weights_v[l-1] / (1 - beta2 ** t)

            #print(self.weights_v[l-1])

            W.data += (lr_w * (m_W_hat / (torch.sqrt(v_W_hat) + eps)))
            #W.data *= (1 - lr_w * 1e-5)

            #print(W.data)






            ###### Update Biases   
            b = self.biases[l-1]
            dB = psi_l


            self.biases_m[l-1] = beta1 * self.biases_m[l-1] + (1-beta1) * dB
            self.biases_v[l-1] = beta2 * self.biases_v[l-1] + (1 - beta2) * (dB.abs() ** 2)

            m_biase_hat = self.biases_m[l-1] / (1 - beta1 ** t)
            v_biase_hat = self.biases_v[l-1] / (1 - beta2 ** t)


            b.data += (lr_b * m_biase_hat / (torch.sqrt(v_biase_hat) + eps))
            #b.data *= (1 - lr_b * 1e-5)


            ###### Updating Gamma 
            ### Update Center
            Gamma = self.centers[l-1]
            ### Combine xi and beta, shape (I[l],)
            xi_beta = xi * beta
            ### Subtract subtract ouput and gamma, shape (I[l], O[l-1])
            delta_center = (y[l-1] - Gamma)  


            dG = (xi_beta.unsqueeze(-1) * delta_center)

            self.centers_m[l-1] = beta1 * self.centers_m[l-1] + (1-beta1) * dG
            self.centers_v[l-1] = beta2 * self.centers_v[l-1] + (1 - beta2) * (dG.abs() ** 2)

            m_center_hat = self.centers_m[l-1] / (1 - beta1 ** t)
            v_center_hat = self.centers_v[l-1] / (1 - beta2 ** t)


            ### Multiply each compounent and update gamma
            Gamma.data += (lr_gamma * m_center_hat / (torch.sqrt(v_center_hat) + eps))
            #Gamma.data *= (1 + lr_gamma * 1e-5)

            ###### Updating sigmas
            ### Collect sigma and prevoius outputs
            sigma = self.sigmas[l-1]
            y_prev = y[l-1]
            ### Subtract and compute Square Euclidean distance, shape (I[l])
            diff_center = y_prev.unsqueeze(-2) - Gamma  
            current_dist_sq = (diff_center.real**2 + diff_center.imag**2).sum(dim=-1)
            dS = (xi * beta * current_dist_sq)


            self.sigmas_m[l-1] = beta1 * self.sigmas_m[l-1] + (1-beta1) * dS
            self.sigmas_v[l-1] = beta2 * self.sigmas_v[l-1] + (1 - beta2) * (dS.abs() ** 2)

            m_sigman_hat = self.sigmas_m[l-1] / (1 - beta1 ** t)
            v_sigmna_hat = self.sigmas_v[l-1] / (1 - beta2 ** t)


            ### Compute multiplication and update sigma weights
            sigma.data += (lr_sigma * m_sigman_hat / (torch.sqrt(v_sigmna_hat) + eps))
            #sigma.data *= (1 - lr_sigma * 1e-5)


            ### Use clamp and avoid exploding activations
            sigma.data.clamp_(min=1e-4)
            
        # Done, now, get error, add azi. and elev., return results back
        loss = (error.real**2 + error.imag**2).sum().item()  
        self.t += 1
        return loss  
    
################## TRAIN AND SET PARAMETERS ##################
###############################################################################################

###############################################################################################
################## COMPUTE SET C-RBF ##################
    def forward(self, x):

        ###### Set intian ouput to x
        y_prev = x  
        
        ####### Iterate through each layer
        for l in range(self.num_layers):
            ###### Get parameters
            W = self.weights[l]      # shape (O[l+1] , I[l+1])
            b = self.biases[l]       # shape (O[l+1],)
            Gamma = self.centers[l]  # shape (I[l+1], O[l])
            sigma = self.sigmas[l]   # shape (I[l+1],)
           
            ###### Compute diff, shape (I[l+1],O[l])
            diff = y_prev.unsqueeze(-2) - Gamma  
            
            ###### Compute Square Euclidean distance, where we only add columns, shape (I[l+1],)
            dist_sq = (diff.real**2 + diff.imag**2).sum(dim=-1)  
            
            ###### Compute phi, becoming real since ||a+jb||^2 = (a^2) + (b^2), shape (I(l+1))
            phi = torch.exp(- dist_sq / (2.0 * sigma))          
            
            ###### Compute y^l = W * phi + b  
            phi_complex = phi.to(dtype=torch.cfloat)
            y = W @ phi_complex + b
          
            ##### Assign new output to the next iteration
            y_prev = y

        ##### Return result, carries two results
        return y_prev 
        
################## COMPUTE SET C-RBF ##################
###############################################################################################

def prediction_plot(model, X_val, true_values, freq_str):
    predicted_azimuth = []
    true_azimuth = []

    predicted_elevation = []
    true_elevation = []

    with torch.no_grad():
        deg = lambda x: x * 180 / torch.pi
        for i in range(0, len(X_val), 5):
            output = model(X_val[i])
            z1, z2 = output[0], output[1]

            predicted_azimuth.append(deg(torch.atan2(z1.imag, z1.real)).item())
            true_azimuth.append(true_values.iloc[i, 0])  # azimuth

            predicted_elevation.append(deg(torch.atan2(z2.imag, z2.real)).item())
            true_elevation.append(true_values.iloc[i, 1])  # elevation

    # Convert to numpy arrays
    true_azimuth = np.array(true_azimuth)
    predicted_azimuth = np.array(predicted_azimuth)

    true_elevation = np.array(true_elevation)
    predicted_elevation = np.array(predicted_elevation)

    store_path = f'Noise_Result/2_Predicted_vs_True_Azimuth_{freq_str}.png'
    plt.figure()
    plt.scatter(true_azimuth, predicted_azimuth, color='blue', alpha=0.6)
    plt.plot([min(true_azimuth), max(true_azimuth)],
            [min(true_azimuth), max(true_azimuth)],
            'r--', label='Perfect Prediction')  # Diagonal reference line
    plt.xlabel('True Azimuth')
    plt.ylabel('Predicted Azimuth')
    plt.title('Predicted vs. True Azimuth')
    plt.legend()
    plt.grid(True)
    plt.savefig(store_path)

    # ===== ELEVATION PLOT (Grouped Mean ± Std) =====
    unique_true = np.unique(true_elevation)
    mean_pred = []
    std_pred = []

    for val in unique_true:
        preds = predicted_elevation[true_elevation == val]
        mean_pred.append(np.mean(preds))
        std_pred.append(np.std(preds))

    mean_pred = np.array(mean_pred)
    std_pred = np.array(std_pred)

    store_path = f'Noise_Result/3_Predicted_vs_True_Elevation_{freq_str}.png'
    plt.figure()
    plt.scatter(true_elevation, predicted_elevation, color='blue', alpha=0.1)
    plt.errorbar(unique_true, mean_pred, 
                 yerr=std_pred, fmt='o', ecolor='gray', capsize=5, color='blue', alpha=0.8)
    plt.plot([min(unique_true), max(unique_true)],
             [min(unique_true), max(unique_true)],
             'r--', label='Perfect Prediction')
    
    plt.scatter(true_elevation, predicted_elevation, color='green', alpha=0.8)
    plt.plot([min(true_elevation), max(true_elevation)],
            [min(true_elevation), max(true_elevation)])  
    
    plt.xlabel('True Elevation')
    plt.ylabel('Mean Predicted Elevation')
    plt.title('Predicted vs. True Elevation (Mean ± Std)')
    plt.legend()
    plt.grid(True)
    plt.savefig(store_path)




class MyDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X_data = X_data
        self.Y_data = Y_data

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        return self.X_data[idx], self.Y_data[idx]


###############################################################################################
################## STARTING POINT ##################
#X_train, Y_train, true_values = get_file('../Normalized_Data_ML_Direction_Finding_selected/Train_Normalized_Model2 - 1.0 Az and 15 Elev Spacing FEKO CSV/train_Normalized_ArrayManifold_UTD_Model2_FEKO_Freq00400MHz_2024_09_11.txt')
#X_val, Y_val, true_values = get_file('../Normalized_Data_ML_Direction_Finding_selected/Val_Normalized_Model2 - 1.0 Az and 15 Elev Spacing FEKO CSV/val_Normalized_ArrayManifold_UTD_Model2_FEKO_Freq00400MHz_2024_09_11.txt')

###### Initialize Input size, Ouput, and hidden layers
input_dim = 8  
output_dim = 2  
hidden_layers = [8,10,8]  # Elements represent number of neurons in each layer

frequencies = [350]  # in MHz, add more as needed
for freq in frequencies:
    
    # Format the frequency as a 5-digit string with leading zeroes (e.g., 00400)
    freq_str = f"{freq:05}"

    # Construct train and val file paths
    train_file_path = f"../Normalized_Data_ML_Direction_Finding_selected/Train_Normalized_Model2 - 1.0 Az and 15 Elev Spacing FEKO CSV/train_Normalized_ArrayManifold_UTD_Model2_FEKO_Freq{freq_str}MHz_2024_09_11.txt"
    val_file_path = f"../Normalized_Data_ML_Direction_Finding_selected/Val_Normalized_Model2 - 1.0 Az and 15 Elev Spacing FEKO CSV/val_Normalized_ArrayManifold_UTD_Model2_FEKO_Freq{freq_str}MHz_2024_09_11.txt"

    
    #val_file_path = f"../Normalized_Data_ML_Direction_Finding_selected/Val_Normalized_Model2 - 1.0 Az and 15 Elev Spacing FEKO CSV/val_Normalized_ArrayManifold_UTD_Model2_FEKO_Freq{freq_str}MHz_2024_09_11.txt"

    print("Hello World")

    # Load the train and validation data
    X_train, Y_train, true_values_train = get_file(train_file_path)
    X_val, Y_val, true_values_val = get_file(val_file_path)

    model = DeepComplexRBF(input_dim, hidden_layers, output_dim)

    ###### Initialize class and set Hyperparameters

    num_epochs = 500
    sample_stride = 1
    learning_rate = 0.0001

    ###### Collect losses to plot in graph later on
    losses_per_epoch = []
    val_losses_per_epoch = []
    testing = 1
    t = 1

    #dataset = MyDataset(X_train, Y_train)
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


    ###### Begin training, iterate number of epochs
    for epoch in range(num_epochs):
        total_loss = 0.0
        #for batch in dataloader:
            #x_sample, y_target = batch

            # Remove batch dimension (since batch_size=1, shape will be [1, ...])
        #    x_sample = x_sample.squeeze(0)
        #    y_target = y_target.squeeze(0)

        for i in range(0, len(X_train), sample_stride):
            x_sample = X_train[i]
            y_target = Y_train[i]
        
            loss = model.training_step(x_sample, y_target, learning_rate, t)
    
            total_loss += loss
        
        t += 1
        avg_loss = total_loss / (len(X_train) // sample_stride)
        losses_per_epoch.append(avg_loss)

        testing += 1


        ################
        ### Validation loss
        val_total_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(X_val), sample_stride):
                x_val = X_val[i]
                y_val = Y_val[i]
                output = model(x_val)
                error = y_val - output
                val_total_loss += (error.real**2 + error.imag**2).sum().item()
        avg_val_loss = val_total_loss / (len(X_val) // sample_stride)
        val_losses_per_epoch.append(avg_val_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        ################
        
    store_path = f'Noise_Result/1_Training_Validation_Loss_FEKO_{freq_str}.png'
    # Plot
    plt.figure()
    plt.plot(range(1, num_epochs + 1), losses_per_epoch, marker='o', markersize=2, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses_per_epoch, marker='s', markersize=2, label='Validation Loss')


    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save it
    plt.savefig(store_path)

    prediction_plot(model, X_val, true_values_val, freq_str)


###### Test NN with parameters results
### Freeze model and stop stracking gradient  
################## STARTING POINT ##################
###############################################################################################


