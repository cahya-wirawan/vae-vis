import torch
import os

# Load the saved PyTorch weights
state_dict = torch.load('decoder_weights.pth', map_location='cpu', weights_only=True)

# Helper function to write a PyTorch tensor into a Rust array
def write_rust_array(file, name, tensor):
    # Flatten the 2D tensors into 1D lists
    vals = tensor.flatten().tolist()
    file.write(f"pub const {name}: [f32; {len(vals)}] = [\n")
    
    # Write in chunks of 10 to keep the file readable and avoid huge lines
    for i in range(0, len(vals), 10):
        chunk = vals[i:i+10]
        file.write("    " + ", ".join(f"{v:.6f}" for v in chunk) + ",\n")
    file.write("];\n\n")

# Make sure the src directory exists
os.makedirs("src", exist_ok=True)

# Generate the Rust source file
print("Exporting weights to src/weights.rs...")
with open("src/weights.rs", "w") as f:
    write_rust_array(f, "W1", state_dict['fc1.weight'])
    write_rust_array(f, "B1", state_dict['fc1.bias'])
    write_rust_array(f, "W2", state_dict['fc2.weight'])
    write_rust_array(f, "B2", state_dict['fc2.bias'])
    write_rust_array(f, "W3", state_dict['fc3.weight'])
    write_rust_array(f, "B3", state_dict['fc3.bias'])

print("✅ Success! Weights baked into Rust source code.")