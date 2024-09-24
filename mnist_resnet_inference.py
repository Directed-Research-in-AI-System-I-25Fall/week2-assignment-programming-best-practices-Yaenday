import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

def main():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the MNIST dataset
    dataset = load_dataset("mnist")
    test_dataset = dataset["test"]

    # Initialize the image processor and model
    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
    
    # Move the model to the appropriate device
    model.to(device)
    model.eval()

    # Define image transformations (including converting to RGB, resizing, and toTensor)
    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB")),  # Convert to RGB
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),  # Convert to Tensor
    ])

    # Preprocess the dataset using the image processor
    def preprocess(example):
        image = example['image']
        image = transform(image)
        # Use image processor to normalize (this will handle the specific model normalization requirements)
        encoding = image_processor(images=image, return_tensors="pt")
        # Squeeze to remove the batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["label"] = example['label']
        return encoding


    test_dataset = test_dataset.map(preprocess, batched=False)

    # Set the format for PyTorch
    test_dataset.set_format(type='torch', columns=['pixel_values', 'label'])

    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize variables for accuracy
    correct = 0
    total = 0

    # Run inference
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            inputs = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Calculate accuracy
    accuracy = (correct / total) * 100
    print(f"Accuracy on MNIST test set: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
