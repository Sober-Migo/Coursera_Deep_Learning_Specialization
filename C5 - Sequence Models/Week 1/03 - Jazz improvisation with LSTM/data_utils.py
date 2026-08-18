from music_utils import * 
from preprocess import * 
from tensorflow.keras.utils import to_categorical

from collections import defaultdict
from mido import MidiFile
from pydub import AudioSegment
from pydub.generators import Sine
import math

#chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
#corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
#N_tones = len(set(corpus))
n_a = 64
x_initializer = np.zeros((1, 1, 86))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def load_music_utils(file):
    chords, abstract_grammars = get_musical_data(file)
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)   
    return (X, Y, N_tones, indices_tones, chords)


def generate_music(inference_model, indices_tones, chords, diversity = 0.5):
    """
    Generates music using a model trained to learn musical patterns of a jazz soloist. Creates an audio stream
    to save the music and play it.
    
    Arguments:
    model -- Keras model Instance, output of djmodel()
    indices_tones -- a python dictionary mapping indices (0-77) into their corresponding unique tone (ex: A,0.250,< m2,P-4 >)
    temperature -- scalar value, defines how conservative/creative the model is when generating music
    
    Returns:
    predicted_tones -- python list containing predicted tones
    """
    
    # set up audio stream
    out_stream = stream.Stream()
    
    # Initialize chord variables
    curr_offset = 0.0                                     # variable used to write sounds to the Stream.
    num_chords = int(len(chords) / 3)                     # number of different set of chords
    
    print("Predicting new values for different set of chords.")
    # Loop over all 18 set of chords. At each iteration generate a sequence of tones
    # and use the current chords to convert it into actual sounds 
    for i in range(1, num_chords):
        
        # Retrieve current chord from stream
        curr_chords = stream.Voice()
        
        # Loop over the chords of the current set of chords
        for j in chords[i]:
            # Add chord to the current chords with the adequate offset, no need to understand this
            curr_chords.insert((j.offset % 4), j)
        
        # Generate a sequence of tones using the model
        _, indices = predict_and_sample(inference_model)
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        
        predicted_tones +=  pred[-1]
                
        #### POST PROCESSING OF THE PREDICTED TONES ####
        # We will consider "A" and "X" as "C" tones. It is a common choice.
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

        # Pruning #1: smoothing measure
        predicted_tones = prune_grammar(predicted_tones)
        
        # Use predicted tones and current chords to generate sounds
        sounds = unparse_grammar(predicted_tones, curr_chords)

        # Pruning #2: removing repeated and too close together sounds
        sounds = prune_notes(sounds)

        # Quality assurance: clean up sounds
        sounds = clean_up_notes(sounds)

        # Print number of tones/notes in sounds
        print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (len([k for k in sounds if isinstance(k, note.Note)]), i))
        
        # Insert sounds into the output stream
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    # Initialize tempo of the output stream with 130 bit per minute
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    # Save audio stream to fine
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("output/my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()
    
    # Play the final stream through output (see 'play' lambda function above)
    # play = lambda x: midi.realtime.StreamPlayer(x).play()
    # play(out_stream)
    
    return out_stream


def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    Ty -- length of the sequence you'd like to generate.
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    ### START CODE HERE ###
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    indices = np.argmax(pred, axis = -1)
    results = to_categorical(indices, num_classes=90)
    ### END CODE HERE ###
    
    return results, indices


def note_to_freq(note, concert_A=440.0):
    """MIDI note number → frequency in Hz (A4 = 440 Hz)."""
    return (2.0 ** ((note - 69) / 12.0)) * concert_A


def _make_rich_tone(freq, duration_ms, volume_db=-18):
    """
    Build a less 'buzzy' tone than a pure sine by mixing a few harmonics
    and applying a short attack + release envelope.
    """
    if duration_ms < 30:
        duration_ms = 30

    # Fundamental + 2nd + 3rd harmonic (quiet) → warmer sound
    fundamental = Sine(freq).to_audio_segment(duration=duration_ms, volume=volume_db)
    h2 = Sine(freq * 2).to_audio_segment(duration=duration_ms, volume=volume_db - 12)
    h3 = Sine(freq * 3).to_audio_segment(duration=duration_ms, volume=volume_db - 18)

    tone = fundamental.overlay(h2).overlay(h3)

    # Short attack / release to kill clicks and pure ringing
    attack = min(25, duration_ms // 4)
    release = min(80, duration_ms // 3)
    tone = tone.fade_in(attack).fade_out(release)
    return tone


def mid2wav(file, out_path="./output/rendered.wav", default_tempo=130):
    """
    Convert a MIDI file to a simple WAV using additive sine synthesis.
    Much better than the original Coursera version (handles note_on velocity=0,
    tempo meta messages, overlapping notes, and uses a richer tone).

    Still not as good as a real SoftSynth + SoundFont, but usable.
    """
    mid = MidiFile(file)

    # Estimate total length (ms)
    total_ms = int(mid.length * 1000) + 500
    output = AudioSegment.silent(duration=total_ms)

    # tempo can change; we keep a running value (µs per quarter note)
    tempo_us = 60_000_000 // default_tempo   # default 130 BPM

    for track in mid.tracks:
        current_tick = 0
        # active notes: channel → note → list of (start_ms, velocity)
        active = defaultdict(lambda: defaultdict(list))

        for msg in track:
            current_tick += msg.time

            # Convert absolute ticks → milliseconds with the *current* tempo
            ms_per_tick = (tempo_us / 1000.0) / mid.ticks_per_beat
            current_pos = current_tick * ms_per_tick

            # Tempo change
            if msg.type == "set_tempo":
                tempo_us = msg.tempo
                continue

            # Note on (velocity > 0) or note off (velocity == 0 or note_off)
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.channel][msg.note].append((current_pos, msg.velocity))
                continue

            if msg.type in ("note_off", "note_on"):  # note_on vel=0 == note_off
                stack = active[msg.channel][msg.note]
                if not stack:
                    continue
                start_pos, velocity = stack.pop()
                duration = max(30, int(current_pos - start_pos))

                # Map MIDI velocity (1-127) to a reasonable dB range
                vol = -30 + (velocity / 127.0) * 18   # roughly -30 dB … -12 dB

                freq = note_to_freq(msg.note)
                rendered = _make_rich_tone(freq, duration, volume_db=vol)
                output = output.overlay(rendered, position=int(start_pos))

    # Soft final limiter-ish
    output = output.normalize(headroom=1.0)
    output.export(out_path, format="wav")
    print(f"Rendered → {out_path}")
    return out_path