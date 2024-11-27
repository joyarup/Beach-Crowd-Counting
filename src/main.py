# src/main.py
from utils.image_loader import load_images_from_folder
from detection.background import create_background_model
from detection.people_detector import process_beach_images
from visualization.visualizer import save_results_with_presentation
from evaluation.metrics import evaluate_detections
from config.default_params import Config
import os

def main():
    # Initialize configuration
    cfg = Config()

    print("Loading images...")
    images, filenames = load_images_from_folder(str(cfg.dataset_folder))

    if not images:
        print("No images found!")
        return

    print("\nCreating background model...")
    background = create_background_model(images)
    
    if background is not None:
        print(f"\nProcessing {len(images)} images...")
        results = process_beach_images(images, background)

        if results:
            print("\nEvaluating detection results...")
            evaluation_metrics = evaluate_detections(
                results, 
                str(cfg.ground_truth_file), 
                filenames
            )
            
            print("\nSaving results...")
            save_results_with_presentation(
                str(cfg.output_folder),
                results,
                images,
                filenames,
                background,
                str(cfg.ground_truth_file)
            )
            
            print("\nEvaluation Results:")
            print(f"Average MSE: {evaluation_metrics['avg_mse']:.2f}")
            print(f"Average Precision: {evaluation_metrics['avg_precision']:.2f}")
            print(f"Average Recall: {evaluation_metrics['avg_recall']:.2f}")
            print(f"Total Matched Detections: {evaluation_metrics['total_matched']}")
            print(f"Total False Positives: {evaluation_metrics['total_false_positives']}")
            print(f"Total False Negatives: {evaluation_metrics['total_false_negatives']}")

            print(f"\nProcessing complete! Results saved in '{cfg.output_folder}'")
        else:
            print("\nNo results to save!")
    else:
        print("\nFailed to create background model!")

if __name__ == "__main__":
    main()