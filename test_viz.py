"""Test script to generate HTML visualization without encoding issues"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.demos.html_viz_v1 import TechnicalJourneyVisualizerHTML

# Redirect print to avoid encoding issues
class SafePrinter:
    def write(self, text):
        try:
            sys.__stdout__.write(text)
        except UnicodeEncodeError:
            sys.__stdout__.write(text.encode('ascii', 'ignore').decode('ascii'))
    def flush(self):
        sys.__stdout__.flush()

sys.stdout = SafePrinter()

try:
    print("Initializing visualizer...")
    visualizer = TechnicalJourneyVisualizerHTML()

    print("Creating HTML processing journey...")
    journey_path = visualizer.create_processing_journey_html(requirement_id="P.9", label="TEST")

    print(f"Success! HTML created at: {journey_path}")
    print("Open in browser to view.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
