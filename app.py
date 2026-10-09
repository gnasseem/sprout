"""Streaming Gradio demo: python app.py --model checkpoints/release"""

import argparse
from pathlib import Path

import gradio as gr
import torch

from inference import FIELDS, load, stream


def build(model, tok):
    def generate(summary, words, sentence, features, temperature, top_k, max_tokens):
        values = (summary, words, sentence, features)
        fields = dict(zip(FIELDS, ("" if value is None else value for value in values)))
        try:
            yield from stream(model, tok, fields, temperature, int(top_k), int(max_tokens))
        except (ValueError, TypeError, OverflowError) as e:
            raise gr.Error(str(e)) from e

    with gr.Blocks(title="Sprout · Story workshop") as demo:
        gr.Markdown("# Sprout\nA small model learning to tell stories. Give it an idea and watch it write.")
        gr.Markdown("Trained from scratch on TinyStories, then instruction tuned. "
                    "It can repeat itself, lose story details, and miss requested words or sentences.")
        with gr.Row():
            with gr.Column():
                summary = gr.Textbox(label="Story idea", value="", lines=3, max_length=4096,
                    placeholder="A little fox plants a seed and learns to be patient.")
                with gr.Accordion("Optional story constraints", open=False):
                    words = gr.Textbox(label="Words to include", value="", placeholder="fox, seed, patient", max_length=4096)
                    sentence = gr.Textbox(label="Sentence to include", value="", placeholder="Good things take time.", max_length=4096)
                    features = gr.Textbox(label="Features", value="", placeholder="Dialogue, MoralValue", max_length=4096)
                with gr.Accordion("Generation settings", open=False):
                    temperature = gr.Slider(0, 2, value=0.8, step=0.05, label="Temperature",
                        info="0 gives the most likely next token; higher values add variety.")
                    top_k = gr.Slider(1, 200, value=50, step=1, label="Top-k")
                    max_tokens = gr.Slider(32, model.ctx, value=256, step=1, label="Maximum new tokens")
                with gr.Row():
                    write = gr.Button("Write a story", variant="primary")
                    stop = gr.Button("Stop")
            with gr.Column():
                story = gr.Textbox(label="Your story", lines=18, interactive=False,
                    placeholder="Your story will appear here, one token at a time.")
        gr.Examples(examples=[
            ["A little fox plants a seed and learns to be patient."],
            ["Two friends find a red box beneath an old tree."],
            ["A tiny dragon learns that blowing bubbles can help others."],
        ], inputs=[summary])
        event = write.click(generate, [summary, words, sentence, features, temperature, top_k, max_tokens],
                            story, api_name="story", concurrency_limit=1)
        stop.click(fn=None, cancels=[event], queue=False)
    return demo.queue(max_size=8, default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    local = Path(__file__).resolve().parent
    bundle = local if (local / "config.json").is_file() else local / "checkpoints/release"
    parser.add_argument("--model", default=str(bundle) if (bundle / "config.json").is_file()
                        else "GeorgeCodesSometimes/sprout-instruct")
    parser.add_argument("--revision", help="Hub commit to load when --model is a model ID")
    parser.add_argument("--host", help="Server bind address; defaults to Gradio's environment setting")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, tok, _ = load(args.model, revision=args.revision)
    build(model, tok).launch(server_name=args.host, server_port=args.port,
                            theme=gr.themes.Default(primary_hue="green"), show_error=True)


if __name__ == "__main__":
    main()
