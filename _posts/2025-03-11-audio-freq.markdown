---
layout: post
title:  "Audio Frequency Explained"
date:   2025-03-11 10:57:10 +0800
categories: audio
---

Original "Hello World":

<div style="display: flex; justify-content: left;">
<audio controls>
    <source src="{{ site.baseurl }}/assets/audio/hello_world.mp3" type="audio/mpeg">
    Your browser does not support the audio element.
</audio>
</div>

Badly stretched "Hello World":

<div style="display: flex; justify-content: left;">
<audio controls>
    <source src="{{ site.baseurl }}/assets/audio/hello_world_bad_stretch.mp3"  type="audio/mpeg">
    Your browser does not support the audio element.
</audio>
</div>

Well stretched "Hello World":

<div style="display: flex; justify-content: left;">
<audio controls>
    <source src="{{ site.baseurl }}/assets/audio/hello_world_good_stretch.mp3" type="audio/mpeg">
    Your browser does not support the audio element.
</audio>
</div>

It can be heard that the badly stretched sound wave is of deep lower voice while the well stretched one preserves the original female voice.

This is attributed to the well stretched sound wave preserves the frequency domain while the badly stretched one simply does interpolation evenly on all sample points.

<div style="display: flex; justify-content: center;">
      <img src="{{ site.baseurl }}/assets/imgs/audio_freq_hello_world.png" width="70%" height="50%" alt="audio_freq_hello_world" />
</div>
