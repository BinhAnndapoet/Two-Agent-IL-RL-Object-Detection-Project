from gymnasium import Env, spaces
import gymnasium as gym


"""
    Detection Environment
    
    The environment is used to define the environment for the SaRLVision agent. Environment inherits from the gymnaisum environment.
"""
class DetectionEnv(Env):
    # Metadata for the environment
    metadata = {"render_modes": ["human", "rgb_array", "bbox", "trigger_image"], "render_fps": 3}
    
    def __init__(self):
        """
            Constructor of the DetectionEnv class.

            Args:
                - env_config: Dictionary that contains the configuration of the environment.
                
            Parameters (env_config):
                - 'dataset': The path of the dataset ('PascalVOC2007_2012Dataset').
                - 'dataset_year': The year of the dataset (2007, 2012, or 2007+2012).
                - 'dataset_image_set': The image set of the dataset (train, val, test).
                - 'obj_configuration': Whether the environment will use single object or multiple objects (0 for single object, 1 for multiple objects).
                - 'current_class': The current class to be used in the environment.
                - 'image': The image to be used in the environment.
                - 'original_image': The original image to be used in the environment.
                - 'target_gt_boxes': The target bounding boxes to be used in the environment.
                - 'target_size': The size of the image that will be used as input to the feature extractor.
                - 'use_sara': Whether the environment will use the SARA model for initial bounding box prediction (True for using the SARA model, False for not using the SARA model).
                - 'feature_extractor': The CNN used to extract the features of the image in the environment.
                - 'max_steps': The maximum number of steps in the environment.
                - 'trigger_steps': The number of steps before the trigger in the environment.
                - 'alpha': The scaling factor for bounding box movements in the environment.
                - 'nu': The trigger reward in the environment.
                - 'threshold': The IoU threshold for the trigger action positive or negative reward in the environment.
                - 'classifier': The CNN used to classify the image ROI in the environment.
                - 'classifier_target_size': The size of the image that will be used as input to the classifier.
                - 'allow_classification': Whether the environment will allow classification or not (True for allowing classification, False for not allowing classification).
                - 'render_mode': The render mode of the environment (None, human, trigger_image, bbox, rgb_array).
                
            Returns:
                - None
        """
        super(DetectionEnv, self).__init__()
        pass

    def train(self):
        """
            Function that sets the environment mode to training.
        """
        pass

    def test(self):
        """
            Function that sets the environment mode to testing.
        """
        
        pass

    def eval(self):
        """
            Function that sets the environment mode to testing.
        """
        
        pass

    def calculate_reward(self):
        """
            Calculating the reward.

            Input:
                - Current state
                - Previous state
                - Target bounding box
                - Reward function

            Output:
                - Reward
        """
        pass
    
    def calculate_trigger_reward(self):
        """
            Calculating the reward.

            Input:
                - Current state
                - Target bounding box
                - Reward function

            Output:
                - Reward
        """
        pass
    
    def get_features(self):
        """
            Getting the features of the image.

            Input:
                - Image
                - Data type

            Output:
                - Features of the image
        """
        pass
    
    def get_state(self):
        """
            Getting the state of the environment.

            Args:
                - dtype: Data type

            Output:
                - State of the environment
        """
        pass
    
    def update_history(self):
        """
            Function that updates the history of the actions by adding the last one.
            It is creating a history vector of size 9, where each element is 0 except the one corresponding to the action performed.
            It is then shifting the history vector by one and adding the new action vector to the history vector.

            Input:
                - Last action performed

            Output:
                - History of the actions
        """
        pass

    def transform_action(self):
        """
            Function that applies the action to the image.

            Actions:
                - 0: Move right
                - 1: Move left
                - 2: Move up
                - 3: Move down
                - 4: Make bigger
                - 5: Make smaller
                - 6: Make fatter
                - 7: Make taller

            Input:
                - Action to apply

            Output:
                - Bounding box of the image
        
        """
        pass
    
    def get_actions(self):
        """
            Function that prints the name of the actions.
        """
        
        pass

    def decode_action(self):
        """
            Function that decodes the action.

            Input:
                - Action to decode

            Output:
                - Unique print of the action
        """
        
        pass

    def rewrap(self):
        """
            Function that rewrap the coordinate if it is out of the image.

            Input:
                - Coordinate to rewrap
                - Size of the image

            Output:
                - Rewrapped coordinate
        """
        pass
    
    def get_info(self):
        """
            Function that returns the information of the environment.

            Output:
                - Information dictionary of the environment
        """
        pass
    
    def generate_random_color(self):
        """
            Function that generates a random color.

            Input:
                - Threshold

            Output:
                - Random color
        """
        pass
    
    def reset(self):
        """
            Function that resets the environment.

            Args:
                - env_config: Dictionary that contains the configuration of the environment.
                - seed: Seed for the environment.
                - options: Options for the environment.
                
            Parameters (env_config):
                - 'image': The image to be used in the environment.
                - 'original_image': The original image to be used in the environment.
                - 'target_bbox': The target bounding box to be used in the environment.
                - 'target_gt_boxes': The target bounding boxes to be used in the environment.
                - 'classifier': The CNN used to classify the image ROI in the environment.
                - 'classifier_target_size': The size of the image that will be used as input to the classifier.
                
            Output:
                - State and information of the environment
        """
        pass
    
    def get_labels(self):
        """
            Function that returns the labels of the images.

            Output:
                - Labels of the images
        """
        pass
    
    def predict(self):
        """
            Function that predicts the label of the image.

            Args:
                - do_display: Whether to display the image or not
                - do_save: Whether to save the image or not
                - save_path: Path to save the image

            Output:
                - Image
        """
        pass

    def restart_and_change_state(self):
        """
            Function that restarts the environment and changes the state.
        """
        pass
        
    def draw_ior_cross(self):
        """
            Function that draws an IoR (Inhibition of Return) cross on the image based on the current bounding box.

            Args:
                - Image
                - Bounding box
                - Color
                - Thickness

            Output:
                - Image with the IoR cross
        """
        pass
    
    def step(self):
        """
            Function that performs an action on the environment.

            Input:
                - Action to perform

            Output:
                - State of the environment
                - Reward of the action
                - Whether the episode is finished or not
                - Information of the environment
        """
        pass
    
    def decode_render_action(self):
        """
        Function that decodes the action.

        Input:
            - Action to decode

        Output:
            - Decoded action as a string
        """
        pass

    def _render_frame(self):
        """
            Function that renders the environment.

            Args:
                - Mode: Mode of rendering (human, trigger_image, bbox, rgb_array)
                - Close: Whether to close the environment or not
                - Alpha: Alpha value for blending the image with the rectangle
                - Text_display: Whether to display the text or not

            Output:
                - Image
        """
        pass

    def render(self):
        """
            Function that renders the environment.

            Args:
                - Mode: Mode of rendering (human, trigger_image, bbox, rgb_array)
                - Close: Whether to close the environment or not

            Output:
                - Image
        """
        pass
    
    def display(self):
        """
            Function that renders the environment.

            Args:
                - Mode: Mode of rendering (image, trigger_image, bbox, detection, heatmap, None)
                - Do_display: Whether to display the image or not
                - Text_display: Whether to display the text or not
                - Alpha: Alpha value for blending the image with the rectangle
                - Color: Color of the bounding box

            Output:
                - Image of the environment
        """
        pass
        
    def segment(self):
        """
            Function that segments the object in the bounding box.

            Note: This function is used for segmentation tasks and is incomplete and not precise as it is a work in progress.

            Args:
                - Display_mode: Mode of display (mask, contour, None)
                - Do_display: Whether to display the image or not
                - Do_save: Whether to save the image or not
                - Save_path: Path to save the image
                - Text_display: Whether to display the text or not
                - Alpha: Alpha value for blending the image with the rectangle
                - Color: Color of the bounding box

            Output:
                - Segmentation dictionary

        """
        pass
    
    def annotate(self):
        """
            Function which utilise the Mask to Annotation software to annotate an object mask in an image.

            Note: This function is a wrapper for the Mask to Annotation software (IEEE ISM 2023) (https://github.com/dylanseychell/mask-to-annotation)

            Args:
                - Image: Image to annotate
                - Id: Id of the image
                - Title: Title of the image
                - Project_name: Name of the project
                - Save_dir: Directory to save the annotation
                - Category: Category of the object
                - Annotation_format: Format of the annotation (coco, vgg, yolo)
                - Do_display: Whether to display the image or not
                - Do_save: Whether to save the image or not
                - Do_print: Whether to print the annotation or not
                - Annotation_color: Color of the annotation
                - Epsilon: Epsilon value for polygon approximation
                - Configuration: Configuration for the annotation
                - Object_configuration: Object configuration for the annotation
                - Do_cvt: Whether to convert the annotation or not
        """
        pass
        
    def plot_img(self):
        """
            Function that plots the image.

            Args:
                - Image: Image to plot
                - Title: Title of the image
                - Figure_size: Size of the figure
        """
        pass
       
    def plot_multiple_imgs(self):
        """
            Function that plots multiple images.

            Args:
                - Images: Images to plot
                - Rows: Number of rows
                - Cols: Number of columns
                - Figure_size: Size of the figure

        """
        
        
    def close(self):
        """
            Function that closes the environment.
        """
        
        pass
    
    def load_pascal_voc_dataset(self):
        """
            Function that loads the Pascal VOC dataset.

            Args:
                - Path: Path to the dataset
                - Year: Year of the dataset (2007, 2012)
                - Download: Whether to download the dataset or not
                - Image_set: Image set of the dataset (train, val, test)

            Output:
                - Dataset
        """
        pass
    
    def load_training_dataset(self):
        """
            Function that loads the Pascal VOC 2007 + 2012 dataset.

            Args:
                - Path: Path to the dataset
                - Download: Whether to download the dataset or not
                - Image_set: Image set of the dataset (train, val, test)

            Output:
                - Dataset
        """
        pass
    
    def sort_pascal_voc_by_class(self):    
        """
            Function that sorts the Pascal VOC dataset by class, by iterating through the dataset and adding the images to the corresponding class.

            Input:
                - Datasets
            
            Output:
                - Dictionary of datasets (keys: classes, values: all the data of this class)
        """
        pass
    
    def extract(self):
        """
            Function that extracts the current image, original image and target bounding box from the dataset.
        """        
        
        pass

    def save_evaluation_results(self):
        """
            Function that saves the evaluation results to a file.

            Args:
                - Path: Path to save the evaluation results
        """
        pass

    def load_evaluation_results(self):
        """
            Function that loads the evaluation results from a file.

            Args:
                - Path: Path to load the evaluation results
        """
        
        pass
    
    def filter_bboxes(self):
        """
            Function that filters the bounding boxes and adds them to the evaluation results.
        """
        pass
        
    def generate_initial_bbox(self):
        """
            Function that generates an initial bounding box prediction based on Saliency Ranking.

            Args:
                - Threshold: Threshold for the Saliency Ranking algorithm
                - Iterations: Number of iterations for the Saliency Ranking algorithm

            Output:
                - Initial bounding box prediction
        """
        pass
    
    def plot_sara(self):
        """
            Function that plots the Saliency Ranking algorithm.

            Args:
                - Threshold: Threshold for the Saliency Ranking algorithm
        """
        pass