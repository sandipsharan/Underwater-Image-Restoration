import collections
import cv2 as cv2
import numpy as np
import scipy as sp
import math
import tensorflow as tf

from matplotlib import pyplot as plt
from skimage import exposure
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, estimate_sigma
from skimage.morphology import closing, disk, square


# Read the input image and predict the depth image using trained model
def readImage(image, size_limit = 256):
    image = cv2.imread(image)
    resized_image = cv2.resize(image,(size_limit,size_limit),interpolation=cv2.INTER_AREA)
    resized_image = cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB)
    resized_image = tf.image.convert_image_dtype(resized_image,tf.float32)
    
    # Loading the model
    model_path = 'C:\\Users\\sandi\\Desktop\\Portfolio_Website\\ENPM673_Project5\\my_model'
    new_model = tf.keras.models.load_model(model_path)
    
    # Depth Prediction
    d_image = data_generation(resized_image, batch=1)
    depth_image = new_model.predict(d_image)
    depth_image = depth_image.squeeze()

    '''To see the Predicted Depth image'''

    # cv2.imshow("Predicted Depth Image", depth_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return np.float32(resized_image)/255.0, np.array(depth_image), resized_image


# Generate data for the model to predict
def data_generation(image, batch, size=256, channels =3):
        x = np.empty((batch, size, size, channels))
        x[0,] = image
        return x


# Preprocess the depth image got from the model
def depth_preprocess(depth_image, min_depth, max_depth):
    z_min = np.min(depth_image) + (min_depth * (np.max(depth_image) - np.min(depth_image)))
    z_max = np.min(depth_image) + (max_depth * (np.max(depth_image) - np.min(depth_image)))
    if max_depth != 0:
        depth_image[depth_image == 0] = z_max
    depth_image[depth_image < z_min] = 0
    return depth_image


# Calculate the points for which backscatter must be estimated using depth ranges
def backscatter_estimation(image,depth_image,fraction=0.01,max_points=30,min_depth_perc=0.0,limit=10):
    z_max, z_min = np.max(depth_image), np.min(depth_image)
    min_depth = z_min + (min_depth_perc * (z_max - z_min))
    z_ranges = np.linspace(z_min,z_max,limit + 1)
    image_norms = np.mean(image, axis=2)
    R_point = []
    G_point = []
    B_point = []
    for i in range(len(z_ranges) - 1):
        a, b = z_ranges[i], z_ranges[i+1]
        indices = np.asarray(np.logical_and(depth_image > min_depth, np.logical_and(depth_image >= a,depth_image <= b ))).nonzero()
        indiced_norm, indiced_image, indiced_depth_image = image_norms[indices], image[indices], depth_image[indices]
        combined_data = sorted(zip(indiced_norm, indiced_image, indiced_depth_image), key=lambda x: x[0])
        points = combined_data[:min(math.ceil(fraction * len(combined_data)), max_points)]
        for _, image_point, depth_image_point in points:
            R_point.append((depth_image_point, image_point[0]))
            G_point.append((depth_image_point, image_point[1]))
            B_point.append((depth_image_point, image_point[2]))
    return np.array(R_point), np.array(G_point), np.array(B_point)


# Calculate the backscatter values for the estimated points
def calculate_backscatter_values(BS_points, depth_image, iterations, max_mean_loss_fraction=0.1):
    BS_depth_val, BS_image_val = BS_points[:, 0], BS_points[:, 1]
    z_max, z_min = np.max(depth_image), np.min(depth_image)
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coeffs = None
    min_loss = np.inf
    lower_limit = [0,0,0,0]
    upper_limit = [1,5,1,5]

    # Calculate the backscatter coefficients
    def estimate_coeffs(depth_image, B_inf, beta_B, J_c, beta_D):
        img_val = (B_inf * (1 - np.exp(-1 * beta_B * depth_image))) + (J_c * np.exp(-1 * beta_D * depth_image))
        return img_val
    
    # Calculate the loss
    def estimate_loss(B_inf, beta_B, J_c, beta_D):
        loss_val = np.mean(np.abs(BS_image_val - estimate_coeffs(BS_depth_val, B_inf, beta_B, J_c, beta_D)))
        return loss_val
    
    for i in range(iterations):
        try:
            est_coeffs, _ = sp.optimize.curve_fit(
                f=estimate_coeffs,
                xdata=BS_depth_val,
                ydata=BS_image_val,
                p0=np.random.random(4) * upper_limit,
                bounds=(lower_limit, upper_limit),
            )
            current_loss = estimate_loss(*est_coeffs)
            if current_loss < min_loss:
                min_loss = current_loss
                coeffs = est_coeffs
        except RuntimeError as re:
            print("Restoring...")

    if min_loss > max_mean_loss:
        print('Applying linear model.', flush=True)
        m, b, _, _, _ = sp.stats.linregress(BS_depth_val, BS_image_val)
        y = (m * depth_image) + b
        return y, np.array([m, b])
    return estimate_coeffs(depth_image, *coeffs), coeffs


# Find the closest non-zero label
def find_closest_label(new_map, begin_x, begin_y):
    t = collections.deque()
    t.append((begin_x, begin_y))
    n_map_mask = np.zeros_like(new_map).astype(bool)
    while not len(t) == 0:
        x, y = t.pop()
        if 0 <= x < new_map.shape[0] and 0 <= y < new_map.shape[1]:
            if new_map[x, y] != 0:
                return new_map[x, y]
            n_map_mask[x, y] = True
            if 0 <= x < new_map.shape[0] - 1:
                x2, y2 = x + 1, y
                if not n_map_mask[x2, y2]:
                    t.append((x2, y2))
            if 1 <= x < new_map.shape[0]:
                x2, y2 = x - 1, y
                if not n_map_mask[x2, y2]:
                    t.append((x2, y2))
            if 0 <= y < new_map.shape[1] - 1:
                x2, y2 = x, y + 1
                if not n_map_mask[x2, y2]:
                    t.append((x2, y2))
            if 1 <= y < new_map.shape[1]:
                x2, y2 = x, y - 1
                if not n_map_mask[x2, y2]:
                    t.append((x2, y2))


# Constructs a neighborhood map using depth and epsilon value
def construct_neighbor_map(depth_image, epsilon=0.05):
    eps = (np.max(depth_image) - np.min(depth_image)) * epsilon
    new_map = np.zeros_like(depth_image).astype(np.int32)
    neighborhoods = 1
    
    #Run the loop until new map has zero value
    while np.any(new_map == 0):
        x_locs, y_locs = np.where(new_map == 0)
        start_index = np.random.randint(0, len(x_locs))
        start_x, start_y = x_locs[start_index], y_locs[start_index]
        que = collections.deque()
        que.append((start_x, start_y))
        while bool(que) == True:
            x, y = que.pop()
            if np.abs(depth_image[x, y] - depth_image[start_x, start_y]) <= eps:
                new_map[x, y] = neighborhoods
                if 0 <= x < depth_image.shape[0] - 1:
                    x2, y2 = x + 1, y
                    if new_map[x2, y2] == 0:
                        que.append((x2, y2))
                if 1 <= x < depth_image.shape[0]:
                    x2, y2 = x - 1, y
                    if new_map[x2, y2] == 0:
                        que.append((x2, y2))
                if 0 <= y < depth_image.shape[1] - 1:
                    x2, y2 = x, y + 1
                    if new_map[x2, y2] == 0:
                        que.append((x2, y2))
                if 1 <= y < depth_image.shape[1]:
                    x2, y2 = x, y - 1
                    if new_map[x2, y2] == 0:
                        que.append((x2, y2))
        neighborhoods += 1
    zeros_arr = sorted(zip(*np.unique(new_map[depth_image == 0], return_counts=True)), key=lambda x: x[1], reverse=True)
    if len(zeros_arr) > 0:
        new_map[new_map == zeros_arr[0][0]] = 0 
    return new_map, neighborhoods - 1


# Refine the constructed neighborhood map 
def refining_neighbor_map(new_map, min_size=10, radius=3):
    pts, counts = np.unique(new_map, return_counts=True)
    neighbor_sizes = sorted([(pt, count) for pt, count in zip(pts, counts)], key=lambda x: x[1], reverse=True)
    refined_new_map = np.zeros_like(new_map)
    total_labels = 1
    for label, size in neighbor_sizes:
        if size >= min_size and label != 0:
            refined_new_map[new_map == label] = total_labels
            total_labels += 1
    for label, size in neighbor_sizes:
        if size < min_size and label != 0:
            for x, y in zip(*np.where(new_map == label)):
                refined_new_map[x, y] = find_closest_label(refined_new_map, x, y)
    refined_n = closing(refined_new_map, square(radius))
    return refined_n, total_labels - 1


# Estimate illumination map from range map using local space average color (LSAC)
def estimate_illumination_map(image, BS_val, neighbor_map, num_neighbor, p=0.5, f=2.0, max_iters=100, tol=1E-5):
    direct_sig = image - BS_val
    avg_space_clr = np.zeros(image.shape)
    avg_space_clr_prime = np.copy(avg_space_clr)
    sizes = np.zeros(num_neighbor)
    indices = [None] * num_neighbor
    for num in range(1, num_neighbor + 1):
        indices[num - 1] = np.where(neighbor_map == num) 
        sizes[num - 1] = np.size(indices[num - 1][0])
    for i in range(max_iters):
        for num in range(1, num_neighbor + 1):
            index = indices[num - 1]
            size = sizes[num - 1] - 1
            avg_space_clr_prime[index] = (1 / size) * (np.sum(avg_space_clr[index]) - avg_space_clr[index])
        new_avg_space_clr = (direct_sig * p) + (avg_space_clr_prime * (1 - p))
        if(np.max(np.abs(avg_space_clr - new_avg_space_clr)) < tol):
            break
        avg_space_clr = new_avg_space_clr
    return f * denoise_bilateral(np.maximum(0, avg_space_clr))


# Filters our redundant data
def data_filter(X, Y, radius_frac=0.01):
    indexes = np.argsort(X)
    X_s = X[indexes]
    Y_s = Y[indexes]
    x_maximum, x_minimum = np.max(X), np.min(X)
    radius = (radius_frac * (x_maximum - x_minimum))
    dS = np.cumsum(X_s - np.roll(X_s, (1,)))
    dX = [X_s[0]]
    dY = [Y_s[0]]
    temp_X = []
    temp_Y = []
    position = 0
    for i in range(1, dS.shape[0]):
        if dS[i] - dS[position] >= radius:
            temp_X.append(X_s[i])
            temp_Y.append(Y_s[i])
            indexes = np.argsort(temp_Y)
            med_idx = len(indexes) // 2
            dX.append(temp_X[med_idx])
            dY.append(temp_Y[med_idx])
            position = i
        else:
            temp_X.append(X_s[i])
            temp_Y.append(Y_s[i])
    return np.array(dX), np.array(dY)


# Calculate beta_d value for an image using depth, illumination and co-efficients
def find_beta_D(depth_image, e, f, g, h):
    return (e * np.exp(f * depth_image)) + (g * np.exp(h * depth_image))


# Calculate Beta_d values
def calculate_wideband_attentuation(depth_image, lumen, r=6, max_value=10.0):
    epsilon = 1E-8
    Beta_D = np.minimum(max_value, -np.log(lumen + epsilon) / (np.maximum(0, depth_image) + epsilon))
    masking = np.where(np.logical_and(depth_image > epsilon, lumen > epsilon), 1, 0)
    refined_atts = denoise_bilateral(closing(np.maximum(0, Beta_D * masking), disk(r)))
    return refined_atts, []


# Refine the calculated Beta_d values
def refining_wideband_attentuation(depth_img, illumination, estimation, iterations=10, min_depth_fraction = 0.1, max_mean_loss_fraction=np.inf, l=1.0, radius_fraction=0.01):
    epsilon = 1E-8
    z_max, z_min = np.max(depth_img), np.min(depth_img)
    min_depth = z_min + (min_depth_fraction * (z_max - z_min))
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coeffs = None
    min_loss = np.inf
    lower_limit = [0, -100, 0, -100]
    upper_limit = [100, 0, 100, 0]
    indices = np.asarray(np.logical_and(illumination > 0, np.logical_and(depth_img > min_depth, estimation > epsilon))).nonzero()
    
    # Calculate the co-efficients
    def calculate_new_rc_depths(depth_img, illum, a, b, c, d):
        E = 1E-5
        result = -np.log(illum + E) / (find_beta_D(depth_img, a, b, c, d) + E)
        return result
    
    # Calulate the loss value
    def calculate_loss(a, b, c, d):
        return np.mean(np.abs(depth_img[indices] - calculate_new_rc_depths(depth_img[indices], illumination[indices], a, b, c, d)))
    
    dX, dY = data_filter(depth_img[indices], estimation[indices], radius_fraction)
    for _ in range(iterations):
        try:
            est_depth_val, _ = sp.optimize.curve_fit(
                f=find_beta_D,
                xdata=dX,
                ydata=dY,
                p0=np.abs(np.random.random(4)) * np.array([1., -1., 1., -1.]),
                bounds=(lower_limit, upper_limit))
            loss = calculate_loss(*est_depth_val)
            if loss < min_loss:
                min_loss = loss
                coeffs = est_depth_val
        except RuntimeError as re:
            print("Restoring....")

    if min_loss > max_mean_loss:
        print('Applying linear model.', flush=True)
        m, b, _, _, _ = sp.stats.linregress(depth_img[indices], estimation[indices])
        beta_d = (m * depth_img + b)
        return l * beta_d, np.array([m, b])
    print(f'Best loss value: {min_loss}', flush=True)
    beta_d = l * find_beta_D(depth_img, *coeffs)
    return beta_d, coeffs


# Scaling the image
def image_scaling(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


# White balance based on top 10% average values of blue and green channel
def image_balancing_10(image):
    green_d = 1.0 / np.mean(np.sort(image[:, :, 1], axis=None)[int(round(-1 * np.size(image[:, :, 0]) * 0.1)):])
    blue_d = 1.0 / np.mean(np.sort(image[:, :, 2], axis=None)[int(round(-1 * np.size(image[:, :, 0]) * 0.1)):])
    sum = green_d + blue_d
    green_d = (green_d/sum)*2.0
    blue_d = (blue_d/sum)*2.0
    image[:, :, 0] *= (blue_d + green_d)/2
    image[:, :, 1] *= green_d
    image[:, :, 2] *= blue_d
    return image


# Reconstruct the scene and globally white balance based the Gray World Hypothesis
def image_restoration(image,depth_image,updated_B,updated_beta_d,new_map):
    restoration = (image - updated_B) * np.exp(updated_beta_d * np.expand_dims(depth_image, axis=2))
    restoration = np.maximum(0.0, np.minimum(1.0, restoration))
    restoration[new_map == 0] = 0
    restoration = image_scaling(image_balancing_10(restoration))
    restoration[new_map == 0] = image[new_map == 0]
    return restoration


# Global Parameters
min_depth = 0.1
max_depth = 1.0
equalization = True

if __name__ == "__main__":

    image_path = "C:\\Users\\sandi\\Desktop\\Portfolio_Website\\ENPM673_Project5\\raw\\raw-890\\41_img_.png"

    resized_image, depth_image, result_image = readImage(image_path)  

    depth_image = depth_preprocess(depth_image,min_depth,max_depth)

    R_back,B_back,G_back = backscatter_estimation(resized_image,depth_image,fraction=0.01,min_depth_perc=min_depth)
    
    BS_red, red_coeffs = calculate_backscatter_values(R_back, depth_image, iterations=25)
    BS_green, green_coeffs = calculate_backscatter_values(G_back, depth_image, iterations=25)
    BS_blue, blue_coeffs = calculate_backscatter_values(B_back, depth_image, iterations=25)

    new_map, _ = construct_neighbor_map(depth_image, 0.1)
    new_map, n = refining_neighbor_map(new_map, 50)

    Red_illumination = estimate_illumination_map(resized_image[:, :, 0], BS_red, new_map, n, p=0.03, f=2.0, max_iters=100, tol=1E-5)
    Green_illumination = estimate_illumination_map(resized_image[:, :, 1], BS_green, new_map, n, p=0.03, f=2.0, max_iters=100, tol=1E-5)
    Blue_illumination = estimate_illumination_map(resized_image[:, :, 2], BS_blue, new_map, n, p=0.03, f=2.0, max_iters=100, tol=1E-5)

    Red_Beta_D, _ = calculate_wideband_attentuation(depth_image, Red_illumination)
    updated_red_beta_d, Red_coefs = refining_wideband_attentuation(depth_image, Red_illumination, Red_Beta_D, radius_fraction=0.05, l=0.5)
    
    Green_Beta_D, _ = calculate_wideband_attentuation(depth_image, Green_illumination)
    updated_green_beta_d, Green_coefs = refining_wideband_attentuation(depth_image, Green_illumination, Green_Beta_D, radius_fraction=0.05, l=0.5)
    
    Blue_Beta_D, _ = calculate_wideband_attentuation(depth_image, Blue_illumination)
    updated_blue_beta_d, Blue_coefs = refining_wideband_attentuation(depth_image, Blue_illumination, Blue_Beta_D, radius_fraction=0.05, l=0.5)

    updated_B = np.stack([BS_red,BS_green,BS_blue],axis=2)
    updated_beta_d = np.stack([updated_red_beta_d,updated_blue_beta_d,updated_green_beta_d],axis=2)
    
    output_image = image_restoration(resized_image,depth_image,updated_B,updated_beta_d,new_map)
    
    if equalization:
        output_image = exposure.equalize_adapthist(np.array(output_image),clip_limit=0.03)
        estimated_sigma = estimate_sigma(output_image, average_sigmas=True, channel_axis=-1)
        output_image = denoise_tv_chambolle(output_image,estimated_sigma)

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(result_image)
    ax[0].set_title("Original Image")
    ax[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Restored Image")
    plt.show()

    print('Image Restored Successfully')