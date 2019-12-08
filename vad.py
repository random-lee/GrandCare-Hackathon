#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Following `a simple but efficient real-time voice activity detection algorithm
<https://www.eurasip.org/Proceedings/Eusipco/Eusipco2009/contents/papers/1569192958.pdf>`__.

There are three criteria to decide if a frame contains speech: energy, most
dominant frequency, and spectral flatness. If any two of those are higher than
a minimum plus a threshold, then the frame contains speech.  In the offline
case, the list of frames is postprocessed to remove too short silence and
speech sequences. In the online case here, inertia is added before switching
from speech to silence or vice versa.
"""

from collections import deque

import numpy as np
import torch
from six.moves import queue

import librosa
import pyaudio
import torchaudio


def compute_spectral_flatness(frame, epsilon=0.01):
    # epsilon protects against log(0)
    geometric_mean = torch.exp((frame + epsilon).log().mean(-1)) - epsilon
    arithmetic_mean = frame.mean(-1)
    return -10 * torch.log10(epsilon + geometric_mean / arithmetic_mean)


class VoiceActivityDetection(object):
    def __init__(
        self,
        num_init_frames=30,
        ignore_silent_count=4,
        ignore_speech_count=1,
        energy_prim_thresh=60,
        frequency_prim_thresh=10,
        spectral_flatness_prim_thresh=3,
        verbose=False,
    ):

        self.num_init_frames = num_init_frames
        self.ignore_silent_count = ignore_silent_count
        self.ignore_speech_count = ignore_speech_count

        self.energy_prim_thresh = energy_prim_thresh
        self.frequency_prim_thresh = frequency_prim_thresh
        self.spectral_flatness_prim_thresh = spectral_flatness_prim_thresh

        self.verbose = verbose

        self.speech_mark = True
        self.silence_mark = False

        self.silent_count = 0
        self.speech_count = 0
        self.n = 0

        if self.verbose:
            self.energy_list = []
            self.frequency_list = []
            self.spectral_flatness_list = []

    def iter(self, frame):

        frame_fft = torch.rfft(frame, 1)
        amplitudes = torchaudio.functional.complex_norm(frame_fft)

        # Compute frame energy
        energy = frame.pow(2).sum(-1)

        # Most dominant frequency component
        frequency = amplitudes.argmax()

        # Spectral flatness measure
        spectral_flatness = compute_spectral_flatness(amplitudes)

        if self.verbose:
            self.energy_list.append(energy)
            self.frequency_list.append(frequency)
            self.spectral_flatness_list.append(spectral_flatness)

        if self.n == 0:
            self.min_energy = energy
            self.min_frequency = frequency
            self.min_spectral_flatness = spectral_flatness
        elif self.n < self.num_init_frames:
            self.min_energy = min(energy, self.min_energy)
            self.min_frequency = min(frequency, self.min_frequency)
            self.min_spectral_flatness = min(
                spectral_flatness, self.min_spectral_flatness
            )

        self.n += 1

        # Add 1. to avoid log(0)
        thresh_energy = self.energy_prim_thresh * torch.log(1.0 + self.min_energy)
        thresh_frequency = self.frequency_prim_thresh
        thresh_spectral_flatness = self.spectral_flatness_prim_thresh

        # Check all three conditions

        counter = 0
        if energy - self.min_energy >= thresh_energy:
            counter += 1
        if frequency - self.min_frequency >= thresh_frequency:
            counter += 1
        if spectral_flatness - self.min_spectral_flatness >= thresh_spectral_flatness:
            counter += 1

        # Detection
        if counter > 1:
            # Speech detected
            self.speech_count += 1
            # Inertia against switching
            if (
                self.n >= self.num_init_frames
                and self.speech_count <= self.ignore_speech_count
            ):
                # Too soon to change
                return self.silence_mark
            else:
                self.silent_count = 0
                return self.speech_mark
        else:
            # Silence detected
            self.min_energy = ((self.silent_count * self.min_energy) + energy) / (
                self.silent_count + 1
            )
            self.silent_count += 1
            # Inertia against switching
            if (
                self.n >= self.num_init_frames
                and self.silent_count <= self.ignore_silent_count
            ):
                # Too soon to change
                return self.speech_mark
            else:
                self.speech_count = 0
                return self.silence_mark
