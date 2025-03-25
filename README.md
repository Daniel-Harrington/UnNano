# UnNano
A GUI Tool for Converting Thorlabs Educational AFM Data to 3D Printable Mesh (.stl)

UnNano enables the conversion of Thorlabs AFM data into 3D printable meshes in ~100ms, providing high-resolution visualizations of 20x20 micrometer samples. Normalization of inputs occurs but relative height differences are preserved.
---


## Usage

1. **Prepare the Directory**:  
   Ensure that the project is located in a folder containing a subfolder named `model_template`.

2. **Install Dependencies**:  
   Install the required dependencies by running the following command:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Add AFM Data**:  
   Place your Thorlabs AFM output file (in semicolon-delimited CSV format) in the designated folder.
---
## Examples
100x100 Pixel Scan:
<p float="left">
  <img src="https://github.com/user-attachments/assets/904d0803-18f5-49d2-843c-04e2eadbbece" alt="scaled_cftest copy" width="200" />
  <img src="https://github.com/user-attachments/assets/5386cc74-b8b6-4ccc-bf03-1775d35bcae4" width="200" /> 
  <img src="https://github.com/user-attachments/assets/e1b7dd49-421b-4466-82df-59fa0759b683" width="200" />
</p>
## Resolution Details

The input CSV file is first converted into a **normalized 32-bit TIFF**, which preserves the relative height accuracy from the AFM data. This TIFF file is then applied to a high-resolution mesh consisting of **652,864 faces**. This mesh has more than sufficient resolution to faithfully represent the height data at the highest AFM resolution of **500x500 pixels**.

- 1 row of pixels in the X-Y dimensions is set to **0** to fuse the mesh with its base layer.
- Blender’s built-in texture interpolation is applied by default to smooth the displacement across the mesh.

