## Cell Counter

A fast, OpenCV-based image analysis tool to count cells in `.vsi` microscope images. Built for neuroscience research in 2 days. Lightweight, accurate enough to perform bulk analysis.

### 🔍 Why

Manual cell counting is slow, tedious, and error-prone. This script automates contour detection and estimates the number of cells in a given image using classical image processing—no deep learning, no black boxes. This script is aimed to process bulk data in an efficient and fast manner with adjustable thresholds and sensitivities.

### ⚙️ How It Works

1. Load `.vsi` image from the `./data` directory
2. Preprocess: grayscale, blur, threshold
3. Detect contours (cell boundaries)
4. Filter + count valid contours
5. Save/overlay result image showing count and boundaries

### 📸 Screenshots

| Original Image                              | Contour Detection                            |
| ------------------------------------------- | -------------------------------------------- |
| ![original](./screenshots/sample_input.png) | ![contours](./screenshots/sample_output.png) |

*Real microscope images used. Count result shown in overlay.*

### 📊 Accuracy

While not pixel-perfect, accuracy is sufficient for practical research needs.

* No advanced separation (yet)
* Best for well-separated, high-contrast images

### 🚀 Usage

```bash
python main.py
```

* Place `.vsi` files inside the `./data` folder
* Results will be saved in the same folder with overlays

### 🧪 Dependencies

* Python 3.11+
* OpenCV

Install with:

```bash
pip install -r requirements.txt
```

(or use `uv`, `poetry`, etc. depending on your setup)

### 🛠️ To Do

- [ ] Add watershed & distance transform for clustered cells
- [ ] CLI parameters for tuning thresholds
- [ ] Batch mode
- [ ] Optional GUI (maybe Streamlit)

### 📄 License

MIT

---

Let’s do this next:

* Send me the screenshots and any filenames to match the placeholders
* If you want, I can write a short blurb for a *mention* in a future LinkedIn or Twitter post

Once that’s in, this thing’s *done and clean*.
