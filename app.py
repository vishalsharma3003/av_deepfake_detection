import gradio as gr
import inference


title="Multimodal deepfake detector"
description="Deepfake detection for videos, images and audio modalities."
            
           
video_interface = gr.Interface(pipeline.deepfakes_video_predict,
                    gr.Video(),
                    "text",
                    examples = ["videos/celeb_synthesis.mp4", "videos/real-1.mp4"],
                    cache_examples = False
                    )


image_interface = gr.Interface(pipeline.deepfakes_image_predict,
                    gr.Image(),
                    "text",
                    examples = ["images/lady.jpg", "images/fake_image.jpg"],
                    cache_examples=False
                    )

audio_interface = gr.Interface(pipeline.deepfakes_audio_predict,
                               gr.Audio(),
                               "text",
                               examples = ["audios/DF_E_2000027.flac", "audios/DF_E_2000031.flac"],
                               cache_examples = False)


app = gr.TabbedInterface(interface_list= [image_interface, video_interface, audio_interface], 
                         tab_names = ['Image inference', 'Video inference', 'Audio inference'])

if __name__ == '__main__':
    app.launch(share = False)