#!/usr/bin/env python3
"""Create images from PDF files.

Reads a directory of PDF files or a single PDF file and saves 
every page as an image.

Images are saved to:
    output/<pdf-stem>/<page>.jpg

where <pdf-stem> is the name of the PDF file without the extension.

If any images are already downloaded, they are skipped (safe to re-run).

Usage
-----
    python pdf_to_images.py 1900-01.pdf
    python pdf_to_images.py pdfs/
    python pdf_to_images.py 1900-01.pdf --dpi 300
    
    python pdf_to_images.py 1900-01.pdf --output-dir output/my-item --dpi 400
"""
import argparse
import sys
from pdf2image import convert_from_path
from pathlib import Path

DEFAULT_DPI = 200
def create_images_from_pdf(pdf_path: Path, output_folder: Path, quiet: bool) -> tuple[int, int]:
    '''Create images from a PDF file.
    Args:
        pdf_path: Path to the PDF file.
        output_folder: Directory to save images into.
        quiet: Whether to suppress progress output.
    Returns:
        A tuple of (created, skipped) image counts.
    '''
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    stem = pdf_path.stem
    page = 1
    num_no_images = 0
    created = 0
    skipped = 0
    while True:
        image_path = output_folder / f"{page}.jpg"
        if image_path.exists():
            if not quiet:
                print(f"Skipping page {page} ({image_path} already exists)", file=sys.stderr)
            skipped += 1
        else:
            
            # Convert one page to an image
            image = convert_from_path(
                pdf_path, 
                DEFAULT_DPI,
                first_page = page,
                last_page = page,
                #poppler_path="C:/Users/mrsmith/Downloads/Release-25.12.0-0/poppler-25.12.0/Library/bin"
            )

            if image:
                image[0].save(image_path, 'JPEG')
                if not quiet:
                    print(f"Saved: {image_path}", file=sys.stderr)
                del image
                created += 1
                num_no_images = 0  # Reset counter if an image was found
            else:
                num_no_images += 1
                if num_no_images >= 3:  # Stop after 3 consecutive pages with no images
                    if not quiet:
                        print("No more images found. Stopping.", file=sys.stderr)
                    break
        page += 1
    return created, skipped
        

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract images from PDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pdf_file",
        nargs="?",
        default=None,
        help="Path to a PDF file (e.g. 1900-01.pdf) or a directory containing PDF files (e.g. pdfs/).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help=(
            "Directory to write images into. "
            "Defaults to output/<pdf-stem>/"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        metavar="DPI",
        help=f"Image DPI (default: {DEFAULT_DPI}). "
        "Higher DPI means higher resolution but slower processing and larger files.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    # Validate: pdf file or directory must be provided and exist.
    if not args.pdf_file:
        parser.print_help(sys.stderr)
        sys.exit(1)
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} does not exist.", file=sys.stderr)
        sys.exit(1)
    
    if pdf_path.is_dir():
        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            print(f"Error: No PDF files found in {pdf_path}.", file=sys.stderr)
            sys.exit(1)
    else:
        if pdf_path.suffix.lower() != ".pdf":
            print(f"Error: {pdf_path} is not a PDF file.", file=sys.stderr)
            sys.exit(1)
        pdf_files = [pdf_path]

    total_downloaded = 0
    total_skipped = 0
    items_processed = 0
    for pdf_file in pdf_files:
        if not args.quiet:
            print(
                f"\n[{items_processed + 1}/{len(pdf_files)}] {pdf_file.stem}",
                file=sys.stderr
            )
        created, skipped = create_images_from_pdf(
            pdf_file, 
            Path(args.output_dir) if args.output_dir else (Path("output") / pdf_file.stem),
            args.quiet
        )

        total_downloaded += created
        total_skipped += skipped
        items_processed += 1

    if not args.quiet:
        print(
            f"\nDone. {items_processed} pdf(s) processed: "
            f"{total_downloaded} images created, {total_skipped} skipped.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
