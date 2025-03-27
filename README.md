# UnNano  
[![Microscopy](https://img.shields.io/badge/Microscopy-AFM-blue.svg)](#)  [![Blender](https://img.shields.io/badge/Blender-3D-orange.svg)](#)  [![Nanomaterials](https://img.shields.io/badge/Nanomaterials-Visualization-green.svg)](#)  [![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](#)

A GUI Tool for Converting Thorlabs Educational AFM Data to 3D Printable Mesh (.stl)

UnNano enables the conversion of Thorlabs AFM data into 3D printable meshes in ~100ms, providing high-resolution visualizations of 20×20 micrometer samples. Normalization of inputs occurs but relative height differences are preserved.
![image](https://github.com/user-attachments/assets/e8f50491-1697-4a2e-acd3-cc3ff3591ad4)

---
## Features
- **Safe removal of even extreme scanline artifacts**
- **Fast Plane Leveling Tool**
- **Adjustable Model Label**
---

## Usage

1. **Prepare the Directory**  
   Ensure that the unnano.py file is located in a folder containing a subfolder named `model_template`.

2. **Install Dependencies**  
   Install the required dependencies by running the following command:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run The Program**
   
   Start unnano.py
   
5. **Add AFM Data**
   
   Drag in a Thorlabs AFM output csv (in semicolon-delimited CSV format)
7. **Generate STL**
   
   Hit the generation button and retrieve your 3D .stl file
---

## Resolution Details

The input CSV file is first converted into a **normalized 32-bit TIFF**, which preserves the relative height accuracy from the AFM data. This TIFF file is then applied to a high-resolution mesh consisting of **652,864 faces**. This mesh has more than sufficient resolution to faithfully represent the height data at the highest AFM resolution of **500×500 pixels**.

- Blender’s built-in texture interpolation is applied by default to smooth the displacement across the mesh.

---

## Examples
100x100 Pixel Scan:
<p float="left">
  <img src="https://github.com/user-attachments/assets/904d0803-18f5-49d2-843c-04e2eadbbece" alt="scaled_cftest copy" width="200" />
  <img src="https://github.com/user-attachments/assets/5386cc74-b8b6-4ccc-bf03-1775d35bcae4" width="200" /> 
  <img src="https://github.com/user-attachments/assets/e1b7dd49-421b-4466-82df-59fa0759b683" width="200" />
</p>
500x500 Pixel Scan:
<p float="left">
  <img src="https://github.com/user-attachments/assets/5f0a064d-282c-46fa-a54b-d2be2d40ddc4" width="200" />
  <img src="https://github.com/user-attachments/assets/75caa05d-dd22-46d9-89e4-654fc58e9a70" width="200" /> 
  <img src="https://github.com/user-attachments/assets/37b160d9-a16b-40b9-848a-ab4e81dcb4f8" width="200" />
</p>
