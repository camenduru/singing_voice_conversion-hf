import gradio as gr


SUPPORTED_TARGET_SINGERS = {
    "Adele": "vocalist_l1_Adele",
    "Beyonce": "vocalist_l1_Beyonce",
    "Bruno Mars": "vocalist_l1_BrunoMars",
    "John Mayer": "vocalist_l1_JohnMayer",
    "Michael Jackson": "vocalist_l1_MichaelJackson",
    "Taylor Swift": "vocalist_l1_TaylorSwift",
    "Jacky Cheung 张学友": "vocalist_l1_张学友",
    "Jian Li 李健": "vocalist_l1_李健",
    "Feng Wang 汪峰": "vocalist_l1_汪峰",
    "Faye Wong 王菲": "vocalist_l1_王菲",
    "Yijie Shi 石倚洁": "vocalist_l1_石倚洁",
    "Tsai Chin 蔡琴": "vocalist_l1_蔡琴",
    "Ying Na 那英": "vocalist_l1_那英",
    "Eason Chan 陈奕迅": "vocalist_l1_陈奕迅",
    "David Tao 陶喆": "vocalist_l1_陶喆",
}


def svc_inference(
    source_audio,
    target_singer,
    diffusion_steps=1000,
    key_shift_mode="auto",
    key_shift_num=0,
):
    pass


demo_inputs = [
    gr.Audio(
        sources=["upload", "microphone"],
        label="Upload (or record) a song you want to listen",
    ),
    gr.Radio(
        choices=list(SUPPORTED_TARGET_SINGERS.keys()),
        label="Target Singer",
        value="Jian Li 李健",
    ),
    gr.Slider(
        1,
        1000,
        value=1000,
        step=1,
        label="Diffusion Inference Steps",
        info="As the step number increases, the synthesis quality will be better while the inference speed will be lower",
    ),
    gr.Radio(
        choices=["Auto Shift", "Key Shift"],
        value="Auto Shift",
        label="Pitch Shift Control",
        info='If you want to control the specific pitch shift value, you need to choose "Key Shift"',
    ),
    gr.Slider(
        -6,
        6,
        value=0,
        step=1,
        label="Key Shift Values",
        info='How many semitones you want to transpose.	This parameter will work only if you choose "Key Shift"',
    ),
]

demo_outputs = gr.Audio(label="")


demo = gr.Interface(
    fn=svc_inference,
    inputs=demo_inputs,
    outputs=demo_outputs,
    title="Amphion Singing Voice Conversion",
)

if __name__ == "__main__":
    demo.launch(show_api=False)
