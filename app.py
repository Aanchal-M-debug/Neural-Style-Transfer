import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io

# Import the architecture needed to load your specific model files
import torch.nn as nn

# ==========================================
# MODEL CLASS DEFINITIONS (Moved from neural_style_network.py)
# ==========================================

class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(128, 64, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(64, 32, kernel_size=3, stride=1),
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4),
        )

    def forward(self, x):
        return self.model(x)

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential()
        current_idx = 0
        
        if upsample:
            self.block.add_module(str(current_idx), nn.Upsample(scale_factor=2, mode='nearest'))
            current_idx += 1
            
        # Reflection Pad
        reflection_padding = kernel_size // 2
        self.block.add_module(str(current_idx), nn.ReflectionPad2d(reflection_padding))
        current_idx += 1
        
        # Conv
        self.block.add_module(str(current_idx), nn.Conv2d(in_channels, out_channels, kernel_size, stride))
        current_idx += 1
        
        # Instance Norm
        if normalize:
            self.block.add_module(str(current_idx), nn.InstanceNorm2d(out_channels, affine=False))
            current_idx += 1
            
        # ReLU
        if relu:
            self.block.add_module(str(current_idx), nn.ReLU(inplace=True))
            current_idx += 1

    def forward(self, x):
        return self.block(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),         # 0
            nn.Conv2d(channels, channels, kernel_size=3, stride=1), # 1
            nn.InstanceNorm2d(channels, affine=False), # 2
            nn.ReLU(inplace=True),         # 3
            nn.ReflectionPad2d(1),         # 4
            nn.Conv2d(channels, channels, kernel_size=3, stride=1), # 5
            nn.InstanceNorm2d(channels, affine=False)  # 6
        )

    def forward(self, x):
        return x + self.block(x)

# ==========================================
# 1. MODEL LOADING LOGIC
# ==========================================

@st.cache_resource
def load_pretrained_model(style_path):
    """Loads YOUR pretrained model from the specific path."""
    with torch.no_grad():
        style_model = TransformerNet()
        try:
            # Attempt to load the weights (state_dict) into the architecture
            state_dict = torch.load(style_path, map_location='cpu')
            style_model.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded model weights from {style_path}")
        except Exception as e:
            # Fallback if the file is a full model save
            print(f"Could not load state_dict, trying full model load: {e}")
            style_model = torch.load(style_path, map_location='cpu')
             
        style_model.eval()
        return style_model

def transform_image(style_model, content_image):
    """Runs the image through the pretrained model to apply style."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess image to be compatible with the model
    # The user's Colab notebook output (Blue Starry Night) vs our "Orange" output suggests
    # we were incorrectly swapping channels to BGR. We will now use standard RGB.
    
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        style_model.to(device)
        output = style_model(content_image).cpu()
        
    # Postprocess output back to image format
    output = output.squeeze(0)
    
    # Robust Auto-Contrast (Percentile-based) to fix "washed out" look
    # We clip the outliers (bottom 1% and top 1%) to stretch the histogram
    output_np = output.numpy()
    
    p1 = np.percentile(output_np, 1)
    p99 = np.percentile(output_np, 99)
    
    if (p99 - p1) > 1e-5:
        output_np = (output_np - p1) / (p99 - p1)
        output_np = output_np * 255.0
    
    # Clamp to valid range
    output_np = np.clip(output_np, 0, 255)
    
    # CHW to HWC
    output_np = output_np.transpose(1, 2, 0)
    output_np = output_np.astype("uint8")
    
    # Saturation Boost (optional but helps match "vibrant" expectations)
    # Convert to HSV, Scaling Saturation, Back to RGB
    try:
        pil_img = Image.fromarray(output_np)
        from PIL import ImageEnhance
        converter = ImageEnhance.Color(pil_img)
        pil_img = converter.enhance(1.3) # 30% boost
        return pil_img
    except:
        return Image.fromarray(output_np)


# ==========================================
# 2. APP UI
# ==========================================

if __name__ == "__main__":
    st.set_page_config(page_title="Neural Style Transfer", layout="centered")

    # --- Professional Pink Theme CSS ---
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
            color: #000000; /* Black text */
        }
        
        .stApp {
            background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%); /* Light Pink Gradient */
        }
        
        h1, h2, h3 {
            color: #000000 !important; /* Black Headers */
            font-weight: 700;
        }
        
        h1 {
            text-align: center;
        }
        
        .stSelectbox label, .stFileUploader label {
            color: #000000 !important; /* Black Labels */
            font-weight: bold;
        }
        
        .stButton>button {
            color: white;
            background-color: #EC407A; /* Vibrant Pink Button */
            border-radius: 8px;
            border: none;
            padding: 12px 28px;
            font-weight: bold;
            transition: 0.3s;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stButton>button:hover {
            background-color: #D81B60;
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            transform: translateY(-2px);
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Neural Style Transfer")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Configuration")
        style_choice = st.selectbox(
            "Select Style Model",
            ("Starry Night", "Pointillism")
        )
        
        if style_choice == "Starry Night":
            model_path = "starry.pth"
        else:
            model_path = "pointillism.pth"
            
    with col2:
        st.subheader("Input Image")
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Original Image', use_container_width=True)
        
        if st.button('Generate Art'):
            with st.spinner('Processing...'):
                try:
                    model = load_pretrained_model(model_path)
                    output_image = transform_image(model, image)
                    
                    st.success("Transformation Complete")
                    st.image(output_image, caption=f'Style: {style_choice}', use_container_width=True)
                    
                    buf = io.BytesIO()
                    output_image.save(buf, format="JPEG")
                    st.download_button("Download Image", buf.getvalue(), "styled_image.jpg", "image/jpeg")
                    
                except FileNotFoundError:
                    st.error(f"Model file {model_path} not found.")
                except Exception as e:
                    st.error(f"Error: {e}")
