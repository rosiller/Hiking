import datetime
from glob import glob
from io import StringIO 
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.cluster import DBSCAN
from time import sleep
import urllib3
import xml.etree.ElementTree as ET

PROCESSED_DIRECTORY = os.path.join(os.path.abspath(''),"1-ProcessedData")

class Hike:
    """
    A class to represent a hiking activity loaded from TCX or GPX files.
    
    Attributes:
    ----------
    filename : str
        Name of the file containing hiking data.
    extension : str
        Extension of the file ('tcx' or 'gpx').
    x : ndarray
        Array of x-coordinates (Longitude) from the file.
    y : ndarray
        Array of y-coordinates (Latitude) from the file.
    datetime : datetime
        Start datetime of the hike.
    date_str : str
        String representation of the hike's date.
    """
    
    def __init__(self, filename, datadir='./2023'):
        """
        Initializes the Hike object with the provided filename and directory.
        
        Parameters:
        ----------
        filename : str
            Name of the file to load.
        datadir : str, optional
            Directory where the file is located. Default is './2023'.
        """
        
        self.filename = filename
        self.extension = filename.split('.')[-1]
        self.x = np.empty(0)
        self.y = np.empty(0)
        
        # Open and read the XML file, correcting any namespace and stripping any extraneous whitespace
        with open(datadir + '/' + self.filename) as xml_file:
            xml_str = xml_file.read().strip()  # Strip needed for possibly corrupted Strava files
            it = ET.iterparse(StringIO(xml_str))
            for _, el in it:
                _, _, el.tag = el.tag.rpartition('}')  # Strip namespace
            root = it.root
            self.root = root  # Temporarily storing the root for further processing
            
        # Extract x, y coordinates and datetime depending on the file extension
        if self.extension == 'tcx': 
            # Processing TCX format
            for child in root.findall('Activities/Activity/Lap/Track/Trackpoint'):
                self.y = np.append(self.y, float(child[1][0].text))
                self.x = np.append(self.x, float(child[1][1].text))
                
            datetime_str = root.findall('Activities/Activity/Id')[0].text
            # Handle timezone formatting
            if ':' == datetime_str[-3:-2]:
                datetime_str = datetime_str[:-3] + datetime_str[-2:]
            self.datetime = datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f%z")
                
        elif self.extension == 'gpx':
            # Processing GPX format
            for trek_point in root.findall('trk/trkseg/trkpt'):
                self.y = np.append(self.y, float(trek_point.items()[0][1]))
                self.x = np.append(self.x, float(trek_point.items()[1][1]))
            self.datetime = datetime.datetime.strptime(root.findall('metadata/time')[0].text, "%Y-%m-%dT%H:%M:%SZ")
        
        # Convert datetime to string for easier display
        self.date_str = self.datetime.strftime('%d %b %Y')

def cluster_hikes(hikes, max_radius, consider_curvature=False, sort=True, verbose=True):
    """
    Cluster hiking trails based on the proximity of their starting points and return the 
    grouped hikes and their bounding boxes.

    Parameters:
    - hikes (list): A list of hike objects.
    - max_radius (float): The maximum distance (in km) between starting points of hikes to be considered in the same cluster.
    - consider_curvature (bool, optional): Whether to consider the curvature of the earth in bounding box calculation. Defaults to False.
    - sort (bool, optional): If True, sorts the hikes within each cluster based on date. Defaults to True.
    - verbose (bool, optional): If True, prints the number of clusters found. Defaults to True.

    Returns:
    - dict: A dictionary where keys are cluster labels and values are lists of hikes belonging to that cluster.
    - dict: A dictionary where keys are cluster labels and values are the bounding boxes of the hikes in that cluster.
    """

    # Extract starting points of all hikes
    start_points = [(hike.x[0], hike.y[0]) for hike in hikes]
    
    # Convert max_radius from kilometers to approximate degrees (assuming 1 degree is roughly 111 km at the equator)
    eps_in_degree = max_radius / 111  
    
    # Cluster hikes using DBSCAN based on proximity of their starting points
    db = DBSCAN(eps=eps_in_degree, min_samples=1).fit(start_points)
    labels = db.labels_
    
    # Group the hikes by their cluster labels
    clustered_hikes = {}
    for idx, label in enumerate(labels):
        if label not in clustered_hikes:
            clustered_hikes[label] = []
        clustered_hikes[label].append(hikes[idx])

    # If verbose mode is on, print the number of clusters identified
    if verbose:
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # -1 in DBSCAN labels indicates noise
        print(f"Number of clusters found: {num_clusters}")

    # Calculate the bounding box for each cluster of hikes
    bounding_boxes = {}
    for label, clustered_hike in clustered_hikes.items():
        min_lon = min([np.min(hike.x) for hike in clustered_hike])
        max_lon = max([np.max(hike.x) for hike in clustered_hike])
        min_lat = min([np.min(hike.y) for hike in clustered_hike])
        max_lat = max([np.max(hike.y) for hike in clustered_hike])

        # If sort flag is set, sort the hikes in a cluster based on their date
        if sort:
            clustered_hikes[label] = sorted(clustered_hike, key=lambda x: datetime.datetime.strptime(x.date_str, '%d %b %Y'))

        # Adjust bounding box to account for curvature of the earth, if the flag is set
        if consider_curvature:
            lon_diff_km = (max_lon - min_lon) * 111 * np.cos(np.radians(min_lat))
            lat_diff_km = (max_lat - min_lat) * 111
            
            # Adjust longitude or latitude bounds depending on which is larger
            if lat_diff_km > lon_diff_km:
                max_side_km = lat_diff_km
                max_side_lon = max_side_km / (111 * np.cos(np.radians(min_lat)))
                max_lon = min_lon + max_side_lon
            else:
                max_side_km = lon_diff_km
                max_side_lat = max_side_km / 111
                max_lat = min_lat + max_side_lat
        else:
            # Ensure the bounding box is square if not considering curvature
            max_side = max(max_lon-min_lon, max_lat-min_lat)
            max_lon = min_lon + max_side
            max_lat = min_lat + max_side

        bounding_boxes[label] = (min_lon, max_lon, min_lat, max_lat)

    return clustered_hikes, bounding_boxes

def crop_image(image_path, output_path, left, top, right, bottom):
    """
    Crop the image with the given margins.
    
    Parameters:
    - image_path (str): Path to the image to crop.
    - output_path (str): Path to save the cropped image.
    - left, top, right, bottom (int): Margins to crop.
    """
    
    with Image.open(image_path) as img:
        width, height = img.size
        cropped_img = img.crop((left, top, width - right, height - bottom))
        cropped_img.save(output_path)

def download_map_image_from_getmap(min_lon, max_lon, min_lat, max_lat, save_directory="",
                                   crop=True, title="", orientation='landscape', return_image=False):
    """
    Help from: https://print.get-map.org/about/api/

    Download a map image from the GET-MAP API based on the provided bounding box.

    Parameters:
    - min_lon, max_lon, min_lat, max_lat (float): Bounding box coordinates.
    - save_directory (str): Directory where the temporary image will be saved.
    - crop (bool): Whether to crop the image.
    - title (str): Title of the map.
    - orientation (str): Either 'landscape' or 'portrait'.
    - return_image (bool): If True, returns a PIL.Image object.

    Returns:
    - PIL.Image or None: If return_image is True, returns a PIL.Image object. Otherwise, returns None.
    """

    if not save_directory:
        print('Please add a save_directory to save the temporal images')
        return
    
    base_url = 'https://api.get-map.org/apis/'
    http = urllib3.PoolManager()

    # Function to perform an API call
    def api_call(url, data=None):
        if data is None:
            r = http.request('GET', url)
        else:
            encoded_data = json.dumps(data).encode('utf-8')
            r = http.request(
                'POST',
                url,
                body=encoded_data,
                headers={'Content-Type': 'application/json'})
        return json.loads(r.data.decode('utf-8'))

    # Specify parameters for the API call
    data = {
        "title": title,
        "bbox_bottom": min_lat,
        "bbox_left": min_lon,
        "bbox_top": max_lat,
        "bbox_right": max_lon,
        "language": "en_US.UTF-8",
        "orientation": orientation,
        "paper_height": 210,
        "paper_width": 210
    }

    r = api_call(base_url + 'jobs/', data)
    job_id = r['id']

    # Poll the job until it's done
    job_status = 0
    while job_status < 2:
        sleep(15)
        r = api_call(base_url + 'jobs/%d' % job_id)
        job_status = r['status']

    # Save the image to a temporary file
    temp_file_name = f"temp_{min_lon}_{max_lon}_{min_lat}_{max_lat}.png"

    r = http.request('GET', r['files']['png'], preload_content=False)
    with open(temp_file_name, 'wb') as out:
        while True:
            data = r.read(4096)
            if not data:
                break
            out.write(data)
    r.release_conn()

    img_copy = None

    # Crop the image if needed
    if crop:
        final_file_name = f"{save_directory}/{min_lon}_{max_lon}_{min_lat}_{max_lat}.png"
        crop_image(temp_file_name, final_file_name, left=62.6, top=62.6, right=62.6, bottom=136.6)
        if return_image:
            with Image.open(final_file_name) as img:
                img_copy = img.copy()
    else:
        os.rename(temp_file_name, f"{save_directory}/{temp_file_name}")

    # Ensure the temporary file is deleted
    if os.path.exists(temp_file_name):
        os.remove(temp_file_name)

    return img_copy

def extend_bbox(bbox, margin_km):
    """
    Extend the bounding box by a given margin in km.
    
    :param bbox: tuple of (min_lon, max_lon, min_lat, max_lat)
    :param margin_km: Margin in kilometers to extend the bounding box by
    :return: Extended bounding box
    """
    min_lon, max_lon, min_lat, max_lat = bbox
    
    # Convert latitude margin from km to degrees
    lat_margin_deg = margin_km / 111.0
    
    # Convert longitude margin from km to degrees. 
    # Using the average latitude for the conversion
    avg_lat = (min_lat + max_lat) / 2.0
    lon_margin_deg = margin_km / (111.0 * np.cos(np.radians(avg_lat)))
    
    # Extend the bbox using the margins
    min_lat -= lat_margin_deg
    max_lat += lat_margin_deg
    min_lon -= lon_margin_deg
    max_lon += lon_margin_deg
        
    return min_lon, max_lon, min_lat, max_lat

def get_and_plot_hikes(data_dir, max_radius=2, margin=0.5):
    """
    Process, cluster, and plot hike data.

    This function retrieves hike data from a directory, clusters the hikes based on 
    their starting points, downloads the corresponding maps from GET-MAP for each cluster,
    and plots the hikes on the maps.

    Parameters:
    - data_dir (str): Directory containing hike data.
    - max_radius (float): The maximum radius (in kilometers) for clustering hikes. Hikes 
      starting within this distance from each other will be clustered together.
    - margin (float): Margin (in degrees) to extend the bounding box of each cluster. This 
      ensures the downloaded map covers a slightly larger area than the cluster itself.

    Returns:
    None
    """

    # Define the directory where processed results will be saved
    save_directory = os.path.join(PROCESSED_DIRECTORY, 'Results')

    # Retrieve hike objects from the data directory
    hike_list = get_hike_objects(data_dir)
    
    # Cluster the hikes based on their starting points
    clustered_hikes, bboxes = cluster_hikes(hike_list, max_radius=max_radius)

    # Loop through each cluster of hikes
    for cluster_label, hike_cluster in clustered_hikes.items():
        
        # Extend the bounding box of the cluster if margin is specified
        if margin:
            bboxes[cluster_label] = extend_bbox(bboxes[cluster_label], margin)

        # Download the map image for the cluster's bounding box
        map_image = download_map_image_from_getmap(*bboxes[cluster_label],
                                        save_directory=save_directory, title="", orientation='landscape', return_image=True)
        
        # Plot the hikes on the map image
        plot_hikes(hike_cluster, bboxes[cluster_label], map_image, save_directory)

def get_hike_objects(data_dir):
    """
    Retrieve a list of Hike objects from specified directory based on the TCX and GPX files present.

    Parameters:
    - data_dir (str): The directory containing the TCX and GPX files.

    Returns:
    - list: A list of Hike objects.
    """
    
    # List to store the names of all relevant files (both TCX and GPX)
    filenames = []

    # Iterate over each file extension and collect all the file names with those extensions
    for extension in ['*.tcx', '*.gpx']:
        filenames.extend([os.path.basename(x) for x in glob(os.path.join(data_dir, extension))])

    # Create Hike objects for each file and add to the list
    hike_list = [Hike(file, datadir=data_dir) for file in filenames]

    return hike_list

def plot_hikes(hike_list, bbox, background_image, save_directory):
    """
    Plots a list of hikes on a given background image, and saves the resulting visualization.

    Parameters:
    - hike_list (list): List of hike objects, where each hike has 'x' and 'y' attributes for its path,
                        and a 'date_str' attribute for its date.
    - bbox (tuple): A tuple containing the bounding box of the area of interest in the format
                    (min_longitude, max_longitude, min_latitude, max_latitude).
    - background_image (str or PIL.Image.Image): Either the path to the background image or a PIL Image object.
    - save_directory (str): Directory where the plot will be saved.

    Returns:
    None
    """
    
    # Prepare the background image
    if isinstance(background_image, Image.Image):
        map_fit = np.array(background_image)  # Convert the PIL.Image object to a NumPy array
    elif isinstance(background_image, str):
        map_fit = plt.imread(background_image)  # Read the image from the given path

    fig, ax = plt.subplots(figsize=(25, 25))

    # Calculate the aspect ratio of the background image
    image_aspect_ratio = map_fit.shape[1] / map_fit.shape[0]  # width/height

    # Calculate the desired change in latitude based on the image's aspect ratio
    delta_lon = bbox[1] - bbox[0]
    desired_delta_lat = delta_lon / image_aspect_ratio
    center_lat = (bbox[2] + bbox[3]) / 2

    # Adjust the bounding box latitudes based on the desired change, centered around the original center latitude
    adjusted_bbox = (bbox[0], bbox[1], center_lat - desired_delta_lat/2, center_lat + desired_delta_lat/2)

    # Display the background image on the plot
    ax.imshow(map_fit, zorder=0, extent=adjusted_bbox, aspect='equal')

    # Plot each hike on the map
    for hike_id, hike in enumerate(hike_list):
        ax.scatter(hike.x, hike.y, zorder=1, alpha=1, c=f'C{hike_id}', s=2, label=hike.date_str)

    # Customize the legend
    lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
    for leghandles in lgnd.legendHandles:
        leghandles._sizes = [200]

    # Hide the axis tick marks
    plt.xticks([])
    plt.yticks([])

    # Save the plot to the specified directory
    plt.savefig(f'{save_directory}/results_{hike_id}')
    plt.show()

