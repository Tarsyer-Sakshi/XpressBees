import os
import tempfile
import streamlit as st
from PIL import Image
import google.generativeai as genai
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy.stats import wasserstein_distance
import torch
from torchvision import models, transforms
import clip
import re
from matplotlib import pyplot as plt


# Ensure compatibility with protobuf 3.20.x

# Set your API key

# Configure the Google AI API with your API key

def version_1():
    # Your version 1 code here
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

        # Dropdown menu for selecting category
        category = st.selectbox("Choose Category", ["Jeans", "Saree", "Kurti", "Other"])

        # Show results button
        show_results = st.button("Show Result")

    with col2:
        if show_results and reference_file is not None and pickup_file is not None:
            if category == "Other":
                st.write("Updating soon")
            else:
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
                    st.write("Pickup image contains plastic or label, upload again.")
                else:
                    st.write("Pickup image is perfect.")

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
                if category == "Jeans" and num_matched > 70:
                    match_status = "Match"
                elif category == "Saree" and num_matched > 180:
                    match_status = "Match"
                elif category == "Kurti" and num_matched > 250:
                    match_status = "Match"
                else:
                    match_status = "Not Matched"

                # Display the matched keypoints image and LBP histogram comparison plot side by side
                st.image([matched_image, lbp_histogram_path], caption=['Matched Keypoints', 'LBP Histogram Comparison'], width=350)
                st.write(f"Visual Anchors: {num_matched}")
                st.write(f"Product Match Distance: {distance}")
                st.write(f"Status: {match_status}")
            
        elif show_results:
            st.write("Please upload both files to continue.")

def version_2():
    # Your version 2 code here
    
    API_KEY = "AIzaSyADFCfEdg0HGtP_zFMyDuqHnwbMmH6RZ-4"

    # Configure the Google AI API with your API key
    genai.configure(api_key=API_KEY)

   



    def upload_to_gemini(path, mime_type=None):
        """Uploads the given file to Gemini and returns the file object."""
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    def compare_images(file1, file2, prompt):
        """Compares two images to check if they are similar using the model."""
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            },
        )
        
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [file1, file2],
                },
            ]
        )
        
        response = chat_session.send_message(prompt)
        
        return response.text

    def process_and_display_images(reference_image, pickup_image, category):
        """
        Display the uploaded reference and pickup images, and show the comparison result based on the category.
        """
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(reference_image.resize((200, 200)), caption='Reference Image', width=200, use_column_width=False)
        with col2:
            st.image(pickup_image.resize((200, 200)), caption='Pickup Image', width=200, use_column_width=False)
        
        if category == "Jeans":
            prompt = "I have to check if both images are similar or not. Allow a little bit of color deviation. Give me response in dict: {'status': True/False, 'explanation': details}"
        elif category == "Kurti / Kurta":
            prompt = "I have to check if both images are similar or not. Allow a little bit of color deviation. Give me response in dict: {'status': True/False, 'explanation': details}"
        elif category == "Saree":
            prompt = "I have to check if both images are similar or not. Allow a little bit of color deviation. Give me response in dict: {'status': True/False, 'explanation': details}"
        elif category == "Top / Shirt / T-shirt":
            prompt = "I have to check if both images are similar or not. Give me response in dict: {'status': True/False, 'explanation': details}"
        else:  # for "Other" category and any other unmatched categories
            prompt = "I have to check if both images are similar or not. Give me response in dict: {'status': True/False, 'explanation': details}"

        result = compare_images(reference_image, pickup_image, prompt)
        
        pattern = r"'status':\s*(True|False),\s*'explanation':\s*'([^']*)'"
        match = re.search(pattern, result)
        if match:
            status = match.group(1) == 'True'
            explanation = match.group(2)
            st.write("Comparison Result:")
            if status:
                st.markdown(f"Status: <span style='color:lightgreen'>Match</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"Status: <span style='color:red'>Not Match</span>", unsafe_allow_html=True)
            st.write("Explanation:", explanation)


    
    col1, col2, col3 = st.columns([1, 0.1, 1])
    with col1:
        st.text("Upload Reference Image")
        reference_image = st.file_uploader("Choose a reference image", type=["jpg", "jpeg", "png"], key="reference")

        st.text("Upload Pickup Image")
        pickup_image = st.file_uploader("Choose a pickup image", type=["jpg", "jpeg", "png"], key="pickup")

        categories = ["Jeans", "Kurti / Kurta", "Saree", "Other"]
        category = st.selectbox("Select Category", categories)

        if st.button("Submit"):
            if reference_image is not None and pickup_image is not None:
                # Create a temporary directory to save uploaded images
                with tempfile.TemporaryDirectory() as tmp_dir:
                    reference_image_path = os.path.join(tmp_dir, "reference_image.jpg")
                    pickup_image_path = os.path.join(tmp_dir, "pickup_image.jpg")

                    # Save the uploaded images to the temporary directory
                    with open(reference_image_path, "wb") as f:
                        f.write(reference_image.getbuffer())
                    with open(pickup_image_path, "wb") as f:
                        f.write(pickup_image.getbuffer())

                    # Open the saved images using PIL
                    reference_image_pil = Image.open(reference_image_path)
                    pickup_image_pil = Image.open(pickup_image_path)

                    # Display the images and result in columns
                    with col3:
                        process_and_display_images(reference_image_pil, pickup_image_pil, category)



def main():
    logo = Image.open('logotarsyer_without_brandname-removebg-preview.png')
    logo_converted = cv2.resize(np.array(logo.convert('RGB')), (100, 100))


    st.set_page_config(page_title="Tarsyer", page_icon=logo, layout="wide")
    logo_image = Image.open('logotarsyer_page-0001-removebg-preview.png')
        # Create columns for the logo and title
    logo_col, title_col = st.columns([0.09, 1])

        # Display the logo in the logo column
    with logo_col:
        st.image(logo_image, width=120)

        # Display the title in the title column
    with title_col:
        st.title('Tarsyer Pickup Assist')    
    version = st.radio("Select Version", ["Version 1", "Version 2"])

    if version == "Version 1":
        version_1()
    elif version == "Version 2":
        version_2()

if _name_ == "_main_":
    main()