# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gradio as gr
import os
import inference

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
    source_audio_path,
    target_singer,
    key_shift_mode="Auto Shift",
    key_shift_num=0,
    diffusion_steps=1000,
):
    #### Prepare source audio file ####
    print("source_audio_path: {}".format(source_audio_path))
    audio_file = source_audio_path.split("/")[-1]
    audio_name = audio_file.split(".")[0]
    source_audio_dir = source_audio_path.replace(audio_file, "")

    ### Target Singer ###
    target_singer = SUPPORTED_TARGET_SINGERS[target_singer]

    ### Inference ###
    if key_shift_mode == "Auto Shift":
        key_shift = "autoshift"
    else:
        key_shift = key_shift_num

    args_list = ["--config", "ckpts/svc/vocalist_l1_contentvec+whisper/args.json"]
    args_list += ["--acoustics_dir", "ckpts/svc/vocalist_l1_contentvec+whisper"]
    args_list += ["--vocoder_dir", "pretrained/bigvgan"]
    args_list += ["--target_singer", target_singer]
    args_list += ["--trans_key", str(key_shift)]
    args_list += ["--diffusion_inference_steps", str(diffusion_steps)]
    args_list += ["--source", source_audio_dir]
    args_list += ["--output_dir", "result"]
    args_list += ["--log_level", "debug"]

    os.environ["WORK_DIR"] = "./"
    inference.main(args_list)

    ### Display ###
    result_file = os.path.join(
        "result/{}/{}_{}.wav".format(audio_name, audio_name, target_singer)
    )
    return result_file


demo_inputs = [
    gr.Audio(
        sources=["upload", "microphone"],
        label="Upload (or record) a song you want to listen",
        type="filepath",
    ),
    gr.Radio(
        choices=list(SUPPORTED_TARGET_SINGERS.keys()),
        label="Target Singer",
        value="Jian Li 李健",
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
    gr.Slider(
        1,
        1000,
        value=1000,
        step=1,
        label="Diffusion Inference Steps",
        info="As the step number increases, the synthesis quality will be better while the inference speed will be lower",
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
    demo.launch()
