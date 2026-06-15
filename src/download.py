import yt_dlp
import torchaudio
import tempfile
import logging

logging.getLogger('yt_dlp').setLevel(logging.CRITICAL)


def download_audio(youtube_id, start_seconds):
    url = f"https://www.youtube.com/watch?v={youtube_id}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'download_ranges': yt_dlp.utils.download_range_func(None, [(start_seconds, start_seconds + 10)]),
            'outtmpl': f'{tmp_dir}/{youtube_id}.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
            'logger': logging.getLogger('yt_dlp'),
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        waveform, sample_rate = torchaudio.load(f"{tmp_dir}/{youtube_id}.wav")

    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = resampler(waveform)
    waveform = waveform.mean(dim=0)

    return waveform.numpy()