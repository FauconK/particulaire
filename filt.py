import numpy as np
import matplotlib.pyplot as plt
from filtrage_particulaire import lecture_image, selectionner_zone, calcul_histogramme

# Parameters
N = 50  # number of particles
Nb = 10  # number of histogram bins
lambda_term = 20  # lambda term for weighting
C1 = C2 = 300  # variances for the normal distribution

# Step 1: Initialize the parameters
std_dev = np.sqrt([C1, C2])

# Step 2: Display the first image and select the tracking zone
first_image, all_image_names, num_images, directory_name = lecture_image(0)
initial_zone = selectionner_zone()
new_im_ref, kmeans_ref, reference_histogram = calcul_histogramme(first_image, initial_zone, Nb)

# Step 3: Initialize the particles
particles = np.random.normal(loc=initial_zone[:2], scale=std_dev, size=(N, 2))
particles = np.hstack((particles, np.tile(initial_zone[2:], (N, 1))))  # Add width and height to each particle

# Step 4: Sequentially read the images from the sequence
for nb_im in range(1, num_images):
    current_image,_,_,_ = lecture_image(nb_im)

    # Step 5: Propagate particles and compute weights
    particles[:, :2] = np.random.normal(particles[:, :2], std_dev.reshape(1, -1))  # Propagate only the x,y position
    weights = np.zeros(N)
    
    for i, particle in enumerate(particles):
        new_im, kmeans,particle_histogram = calcul_histogramme(current_image,particle, Nb)
        weights[i] = np.exp(-lambda_term * np.sum((reference_histogram - particle_histogram) ** 2))

    # Step 6: Normalize the weights and resample
    weights /= np.sum(weights)  # Normalize
    indices = np.random.choice(range(N), size=N, p=weights)
    particles = particles[indices]

    # Step 7: Estimate the position
    estimated_state = np.average(particles, weights=weights, axis=0)

    # Step 8: Display the image with the estimated rectangle and particle positions
    plt.imshow(current_image)
    plt.gca().add_patch(plt.Rectangle((estimated_state[0], estimated_state[1]), estimated_state[2], estimated_state[3], edgecolor='r', facecolor='none'))
    plt.scatter(particles[:, 0], particles[:, 1], c='b', marker='o')
    plt.show()

    print(f"Estimated State: {estimated_state}")

# The code assumes that the 'lecture_image' function returns the current image to be processed.
# The actual image processing and display logic is not included in this snippet.
