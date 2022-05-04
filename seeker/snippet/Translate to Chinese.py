#date: 2022-05-04T17:17:48Z
#url: https://api.github.com/gists/071da86e5d705da425bc7ee9bf17814c
#owner: https://api.github.com/users/insightsbees

if choose == "Chinese":
    if st.button("Show Translation and Audio"):
        input='en'                   
        output='zh-cn'
        audio_file, translation_text = translation_func(input, output, text)

        st.write('  ')
        segments = jieba.cut(translation_text) #used for chinese text segmentation
        seg_output = " ".join(segments)
        html_str = f"""
        <style>
        p.a {{
        font: bold {30}px Courier;
        }}
        </style>
        <p class="a">{seg_output}</p>
        """
        st.markdown(html_str, unsafe_allow_html=True) #display the translated text in Chinese characters
        st.write('  ')

        p = Pinyin()
        pinyined = p.get_pinyin(seg_output, splitter='', tone_marks='marks') #Get pinyin (the official romanization system for Standard Chinese in mainland China)
        html_str2 = f"""
        <style>
        p.a {{
        font: bold {25}px Courier;
        }}
        </style>
        <p class="a">{pinyined}</p>
        """
        st.markdown(html_str2, unsafe_allow_html=True) #display pin yin

        audio_file = open(f"temp_folder/{audio_file}.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3", start_time=0) #displays the audio player