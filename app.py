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


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Amphion Singing Voice Conversion: *DiffWaveNetSVC*
        [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/usercenter/Amphion)        
        This demo provides an Amphion [DiffWaveNetSVC](https://github.com/open-mmlab/Amphion/tree/main/egs/svc/MultipleContentsSVC) pretrained model for you to play. The training data has been detailed [here](https://huggingface.co/amphion/singing_voice_conversion).
        """
    )

    gr.Markdown(
        """
        ## Source Audio
        **Hint**: We recommend using dry vocals (e.g., studio recordings or source-separated voices from music) as the input for this demo. At the bottom of this page, we provide some examples for your reference.
        """
    )
    source_audio_input = gr.Audio(
        sources=["upload", "microphone"],
        label="Source Audio",
        type="filepath",
    )

    with gr.Row():
        with gr.Column():
            config_target_singer = gr.Radio(
                choices=list(SUPPORTED_TARGET_SINGERS.keys()),
                label="Target Singer",
                value="Jian Li 李健",
            )
            config_keyshift_choice = gr.Radio(
                choices=["Auto Shift", "Key Shift"],
                value="Auto Shift",
                label="Pitch Shift Control",
                info='If you want to control the specific pitch shift value, you need to choose "Key Shift"',
            )

        # gr.Markdown("## Conversion Configurations")
        with gr.Column():
            config_keyshift_value = gr.Slider(
                -6,
                6,
                value=0,
                step=1,
                label="Key Shift Values",
                info='How many semitones you want to transpose.	This parameter will work only if you choose "Key Shift"',
            )
            config_diff_infer_steps = gr.Slider(
                1,
                1000,
                value=1000,
                step=1,
                label="Diffusion Inference Steps",
                info="As the step number increases, the synthesis quality will be better while the inference speed will be lower",
            )
            btn = gr.ClearButton(
                components=[
                    config_target_singer,
                    config_keyshift_choice,
                    config_keyshift_value,
                    config_diff_infer_steps,
                ]
            )
            btn = gr.Button(value="Submit", variant="primary")

    gr.Markdown("## Conversion Result")
    demo_outputs = gr.Audio(label="Conversion Result")

    btn.click(
        fn=svc_inference,
        inputs=[
            source_audio_input,
            config_target_singer,
            config_keyshift_choice,
            config_keyshift_value,
            config_diff_infer_steps,
        ],
        outputs=demo_outputs,
    )

    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            [
                "examples/chinese_female_recordings.wav",
                "John Mayer",
                "Auto Shift",
                1000,
                "examples/output/chinese_female_recordings_vocalist_l1_JohnMayer.wav",
            ],
            [
                "examples/chinese_male_seperated.wav",
                "Taylor Swift",
                "Auto Shift",
                1000,
                "examples/output/chinese_male_seperated_vocalist_l1_TaylorSwift.wav",
            ],
            [
                "examples/english_female_seperated.wav",
                "Feng Wang 汪峰",
                "Auto Shift",
                1000,
                "examples/output/english_female_seperated_vocalist_l1_汪峰.wav",
            ],
            [
                "examples/english_male_recordings.wav",
                "Yijie Shi 石倚洁",
                "Auto Shift",
                1000,
                "examples/output/english_male_recordings_vocalist_l1_石倚洁.wav",
            ],
        ],
        inputs=[
            source_audio_input,
            config_target_singer,
            config_keyshift_choice,
            config_diff_infer_steps,
            demo_outputs,
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True)
