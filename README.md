# Network Visualization App
![image](https://github.com/user-attachments/assets/9413e95d-4192-4f8c-aff0-5ab642ba6689)

A PyQt-based desktop application for loading, visualizing, and analyzing network (graph) data using a competitive dynamics model.

---

## Requirements

Make sure to install the required Python packages:

```bash
pip install pandas numpy torch networkx pyqt5 pyqtgraph
```

---

## How to Run

```bash
python path/to/app.py
```

---

Alternatively, you can download the built app here: (https://drive.google.com/drive/folders/1N4_b5NjFDvr01jKEMzQQYaWjiT2FdDed?usp=sharing) and run App.exe

## Interface Overview

The GUI consists of:

- **Left Panel**:
  - Load data
  - Select columns for source, target, etc.
  - Run model
  - View and save results
- **Right Panel**:
  - Network graph visualization
  - Plot showing convergence (`delta` per iteration)

---

## 🛠 Usage Instructions

### 🔹 Step 1: Load Network File

- Go to the menu **File → Load network from file**.
- Choose a file (`.csv`, `.tsv`, `.txt`) that contains your graph data.
- The file must have at least `Start` and `End` node columns.

### 🔹 Step 2: Select Columns

Use the combo boxes to map your data:
- **Start**: Source node column
- **End**: Target node column
- **Weight** *(optional)*: Edge weight (defaults to 1 if omitted)
- **Direct** *(optional)*: Direction flag (1 = directed, 0 = undirected)
- **Network Type**: 
  - `Directed`
  - `Undirected`
  - `Mixed`

### 🔹 Step 3: Load Network

Click the **Load Network** button. The app will:
- Render the graph using a force-directed layout
- Prepare internal matrices for analysis

### 🔹 Step 4: Run Analysis

Click **Execute** to:
- Run the inside competitive dynamics model
- Visualize the convergence (Delta per Iteration plot)

### 🔹 Step 5: Save Results

Click **Save Result** to export the output table as `.xlsx`.

---

## Input File Example

Here’s an example of a valid input file:

```csv
Source,Target,Weight,Direction
A,B,0.5,1
B,C,1.2,0
C,D,0.8,1
```

---

## Notes

- If `Mixed` network type is selected, a **Direct** column must be provided.
