from utils.image_loader import load_images_from_folder
from detection.background import create_background_model
from detection.people_detector import process_beach_images
from visualization.visualizer import save_results_with_presentation
from evaluation.metrics import evaluate_detections
import os

def main():
    dataset_folder = "Dataset"
    output_folder = "detection_results"
    ground_truth_file = "labels_irving-arup_2024-11-26-08-18-48.csv"

    print("Loading images...")
    images, filenames = load_images_from_folder(dataset_folder)

    if not images:
        print("No images found!")
        return

    print(f"\nProcessing {len(images)} images...")
    results, background = process_beach_images(images)

    if results:
        print("\nEvaluating detection results...")
        evaluation_metrics = evaluate_detections(results, ground_truth_file, filenames)
        
        print("\nSaving results...")
        save_results_with_presentation(
            output_folder,
            results,
            images,
            filenames,
            background,
            ground_truth_file
        )
        
        print("\nEvaluation Results:")
        print(f"Average MSE: {evaluation_metrics['avg_mse']:.2f}")
        print(f"Average Precision: {evaluation_metrics['avg_precision']:.2f}")
        print(f"Average Recall: {evaluation_metrics['avg_recall']:.2f}")
        print(f"Total Matched Detections: {evaluation_metrics['total_matched']}")
        print(f"Total False Positives: {evaluation_metrics['total_false_positives']}")
        print(f"Total False Negatives: {evaluation_metrics['total_false_negatives']}")

        print(f"\nProcessing complete! Results saved in '{output_folder}'.")
    else:
        print("\nNo results to save!")

if __name__ == "__main__":
    main()