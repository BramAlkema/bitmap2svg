# bitmap2svg/bitmap2svg/README.md

# Bitmap2SVG

Bitmap2SVG is a hybrid Potrace and geometry-based vectorization pipeline designed to convert flat-color logos into scalable vector graphics (SVG). This project leverages various image processing techniques, including k-means clustering, Ramer-Douglas-Peucker simplification, and swarm optimization, to produce high-quality vector representations of bitmap images.

## Features

- **Image Ingestion**: Load and normalize images in various formats.
- **Segmentation**: Segment images into layers using k-means clustering.
- **Vectorization**: Convert bitmap images to vector paths using Potrace.
- **Simplification**: Simplify polylines with the Ramer-Douglas-Peucker algorithm.
- **Bezier Fitting**: Fit cubic Bézier curves to polylines for smoother paths.
- **Quality Assurance**: Evaluate generated SVGs against original images using metrics like SSIM and edge IoU.
- **Optional OCR**: Extract text from images using optical character recognition.
- **FastAPI Service**: Serve the vectorization functionality over HTTP.
- **Command-Line Interface**: Easily vectorize images and process batches from the command line.

## Installation

To install the required dependencies, run:

```bash
pip install -U pip
pip install -e .
```

Make sure to have the necessary libraries installed, including `numpy`, `opencv-python`, `Pillow`, `svgwrite`, `shapely`, `pyclipper`, `rdp`, `pydantic`, `typer`, `rich`, and `potrace-wrapped`.

## Usage

### Command-Line Interface

You can use the command-line interface to vectorize images:

```bash
python -m bitmap2svg.cli vectorise path/to/logo.png -o out.svg
```

### Batch Processing

To process a batch of images, use:

```bash
python -m bitmap2svg.cli batch path/to/images/ -o output_directory/
```

### FastAPI Service

To run the FastAPI service, execute:

```bash
uvicorn bitmap2svg.service:app --reload
```

## Testing

To run the tests, use:

```bash
pytest
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.