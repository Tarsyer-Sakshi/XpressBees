import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.stats import wasserstein_distance
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import clip

# Set page configuration
logo = Image.open('D:\\Tarsyer\\4_Xpressbees\\logotarsyer_without_brandname-removebg-preview.png')
logo_converted = cv2.resize(np.array(logo.convert('RGB')), (100, 100))
st.set_page_config(page_title="Tarsyer", page_icon=logo, layout="wide")
 
# Title of the web app
logo_image = Image.open('D:\\Tarsyer\\4_Xpressbees\\logotarsyer_page-0001-removebg-preview.png')
# Create columns for the logo and title
logo_col, title_col = st.columns([0.09, 1])

# Display the logo in the logo column
with logo_col:
    st.image(logo_image, width=120)
# Display the title in the title column
with title_col:
    st.title('Tarsyer Pickup Assist')

# Custom CSS to add a 10px gap between columns
st.markdown(
    """
    <style>
    .block-container {
        display: flex;
    }
    .block-container .block {
        flex: 1;
    }
    .block-container .spacer {
        width: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create columns with a spacer in between
col1, spacer, col2 = st.columns([1, 0.1, 1])

def load_deeplabv3_model():
    model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
    return model

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)

def segment_image(model, input_image):
    with torch.no_grad():
        output = model(input_image)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions

def remove_background(image):
    # Load model
    model = load_deeplabv3_model()

    # Preprocess image
    input_image = preprocess_image(image)

    # Segment image
    mask = segment_image(model, input_image)

    # Convert PIL image to OpenCV format
    original_image = np.array(image)
    original_size = original_image.shape[:2]

    # Resize mask to original image size
    mask = cv2.resize(mask.byte().cpu().numpy(), (original_size[1], original_size[0]))

    # Create a binary mask where the person is labeled (typically label 15 for 'person' in COCO dataset)
    person_label = 15
    binary_mask = np.where(mask == person_label, 1, 0).astype(np.uint8)

    # Apply mask to the original image
    foreground = cv2.bitwise_and(original_image, original_image, mask=binary_mask)

    # Convert the image to RGB format
    foreground_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)

    return Image.fromarray(foreground_rgb)

def compute_lbp_histogram(image, P=8, R=1):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def compute_results(ref_image, pickup_image):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT keypoints and descriptors in both images
    keypoints_ref, descriptors_ref = sift.detectAndCompute(ref_image, None)
    keypoints_pickup, descriptors_pickup = sift.detectAndCompute(pickup_image, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.match(descriptors_ref, descriptors_pickup)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw top matches
    matched_image = cv2.drawMatches(ref_image, keypoints_ref, pickup_image, keypoints_pickup, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save the matched keypoints image
    cv2.imwrite('matched_keypoints.jpg', matched_image)

    # Compute LBP histograms for the images
    lbp_hist_ref = compute_lbp_histogram(ref_image)
    lbp_hist_pickup = compute_lbp_histogram(pickup_image)

    # Compare LBP histograms using the Wasserstein distance
    distance = wasserstein_distance(lbp_hist_ref, lbp_hist_pickup)

    # Plot the LBP histograms
    plt.figure(figsize=(6, 3))
    plt.bar(range(len(lbp_hist_ref)), lbp_hist_ref, alpha=0.7, label='Reference Image')
    plt.bar(range(len(lbp_hist_pickup)), lbp_hist_pickup, alpha=0.7, label='Pickup Image')
    plt.title(f"LBP Histogram Comparison\nWasserstein Distance: {distance:.3f}")
    plt.legend()

    # Save the LBP histogram comparison plot
    plt.savefig('lbp_histogram_comparison.jpg')

    return matched_image, 'matched_keypoints.jpg', 'lbp_histogram_comparison.jpg', len(matches), distance

with col1:
    # Upload button for reference file
    reference_file = st.file_uploader('Upload Reference File', type=['png', 'jpeg', 'jpg'])

    # Upload button for pickup file
    pickup_file = st.file_uploader('Upload Pickup File', type=['png', 'jpeg', 'jpg'])

    # Show results button
    show_results = st.button("Show Result")

with col2:
    if show_results and reference_file is not None and pickup_file is not None:
        # Preprocessing code
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        # Read and preprocess images
        pickup_image_pil = Image.open(pickup_file)
        reference_image_pil = Image.open(reference_file)

        # Check for plastic or label in pickup image
        pickup_image_tensor = preprocess(pickup_image_pil).unsqueeze(0).to(device)
        text = clip.tokenize(["Plastic", "No Plastic", "Label", "No Label"]).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(pickup_image_tensor)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(pickup_image_tensor, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        if probs[0][0] > 0.5 or probs[0][2] > 0.5:
            st.write("Pickup image contains plastic or label upload again.")
        else:
            st.write("Pickup image is perfect.")

        # Classification code for reference image
        reference_image_tensor = preprocess(reference_image_pil).unsqueeze(0).to(device)
        text2 = clip.tokenize(["Jeans", "Saree", "Kurti"]).to(device)
        labels = ["Jeans", "Saree", "Kurti"]

        with torch.no_grad():
            image_features = model.encode_image(reference_image_tensor)
            text_features = model.encode_text(text2)

            logits_per_image, logits_per_text = model(reference_image_tensor, text2)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        predicted_label_idx = probs.argmax()
        predicted_label = labels[predicted_label_idx]
        st.write(f"Category: {predicted_label}")

        # Remove background from the reference image
        reference_image_no_bg = remove_background(reference_image_pil)

        # Convert images to RGB
        reference_image_rgb = cv2.cvtColor(np.array(reference_image_no_bg), cv2.COLOR_RGB2BGR)
        pickup_image_rgb = pickup_image_pil.convert('RGB')

        # Convert images to numpy arrays
        pickup_image_np = np.array(pickup_image_rgb)
        
        # Compute SIFT matches and LBP histograms
        matched_image, matched_keypoints_path, lbp_histogram_path, num_matched, distance = compute_results(reference_image_rgb, pickup_image_np)

        # Determine match status based on the classified data
        if predicted_label == "Jeans" and num_matched > 70:
            match_status = "Match"
        elif predicted_label == "Saree" and num_matched > 180:
            match_status = "Match"
        elif predicted_label == "Kurti" and num_matched > 250:
            match_status = "Match"
        else:
            match_status = "Not Matched"
        
        # Display the matched keypoints image and LBP histogram comparison plot side by side
        st.image([matched_image, lbp_histogram_path], caption=['Matched Keypoints', 'LBP Histogram Comparison'], width=350)
        st.write(f"Keypoints matched: {num_matched}")
        st.write(f"Wasserstein Distance: {distance}")
        st.write(f"Status: {match_status}")
        
    elif show_results:
        st.write("Please upload both files to continue.")
