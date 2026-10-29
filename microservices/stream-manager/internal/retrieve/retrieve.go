// SPDX-FileCopyrightText: Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Package retrieve cuts frames and clips out of ready or live recordings.
package retrieve

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/open-edge-platform/edge-ai-libraries/microservices/stream-manager/internal/model"
	"github.com/open-edge-platform/edge-ai-libraries/microservices/stream-manager/internal/storage"
)

const (
	maxConcurrentExtractions = 6
	// exactTolerance is how far a frame may sit from the requested time and
	// still count as an exact match.
	exactTolerance = time.Millisecond
)

var (
	ErrInvalidRequest = errors.New("invalid retrieve request")
	ErrNotFound       = errors.New("recording not found")
	ErrNotReady       = errors.New("recording is not ready")
	ErrNotCovered     = errors.New("requested time is not covered by the recording")
	ErrFrameNotFound  = errors.New("no frame sits at the requested time")
	// ErrBeyondSearchLimit means the frame or its keyframe is farther than SearchLimit from the requested time.
	ErrBeyondSearchLimit = errors.New("no decodable frame within the search limit")
	ErrMediaNotFound     = errors.New("media not found")
	ErrBusy              = errors.New("extraction capacity reached")
	ErrExtraction        = errors.New("media extraction failed")
)

var contentTypes = map[string]string{"jpeg": "image/jpeg", "png": "image/png", "mp4": "video/mp4"}

type Retriever interface {
	Frame(ctx context.Context, recordingID string, timestamp time.Time, format, match string) (Media, error)
	Clip(ctx context.Context, recordingID string, startTS time.Time, duration time.Duration, format string) (Media, error)
	OpenMedia(ctx context.Context, recordingID, name string) (*os.File, MediaFile, error)
}

// `Media` is a extracted frame or a cut clip (aka derived media).
// `Name` is the file name under the recording's derived directory and `Path` its absolute location.
type Media struct {
	Result     model.MediaResult
	Name       string
	Path       string
	ExactMatch bool
}

// MediaFile describes an opened derived media (frame or clip).
type MediaFile struct {
	ContentType string
	ExpiryTS    *time.Time
}

// Service implements Retriever with FFmpeg and FFprobe.
type Service struct {
	metadata storage.MetadataHandler
	media    storage.MediaHandler
	ffmpeg   string
	ffprobe  string
	slots    chan struct{}
	now      func() time.Time
}

var _ Retriever = (*Service)(nil)

func NewService(metadata storage.MetadataHandler, media storage.MediaHandler) (*Service, error) {
	ffmpeg, err := exec.LookPath("ffmpeg")
	if err != nil {
		return nil, fmt.Errorf("find FFmpeg: %w", err)
	}
	ffprobe, err := exec.LookPath("ffprobe")
	if err != nil {
		return nil, fmt.Errorf("find FFprobe: %w", err)
	}
	return &Service{
		metadata: metadata, media: media, ffmpeg: ffmpeg, ffprobe: ffprobe,
		slots: make(chan struct{}, maxConcurrentExtractions), now: time.Now,
	}, nil
}

// Frame returns the frame nearest to timestamp, or the frame at exactly that
// time when match is "exact".
func (s *Service) Frame(ctx context.Context, id string, ts time.Time, format, match string) (Media, error) {
	_, ok := contentTypes[format]
	if !ok || format == "mp4" || (match != "nearest" && match != "exact") {
		return Media{}, ErrInvalidRequest
	}
	rec, err := s.retrievableRecording(ctx, id)
	if err != nil {
		return Media{}, err
	}
	if ts.Before(rec.StartTS) || (rec.EndTS != nil && !ts.Before(*rec.EndTS)) {
		return Media{}, ErrNotCovered
	}
	release, err := s.acquire()
	if err != nil {
		return Media{}, err
	}
	defer release()
	tl, releaseMedia, err := s.openTimeline(ctx, rec)
	if err != nil {
		return Media{}, err
	}
	defer releaseMedia()
	frame, keyframe, err := tl.nearestFrame(ctx, ts)
	if err != nil {
		return Media{}, err
	}
	frameTS := tl.wall(frame.PTS)
	if rec.State == "recording" {
		hasLaterFrame, err := tl.hasFrameAfter(ctx, ts)
		if err != nil {
			return Media{}, err
		}
		if !hasLaterFrame && !frameTS.Equal(ts) {
			return Media{}, ErrNotCovered
		}
	}
	exact := absDuration(frameTS.Sub(ts)) <= exactTolerance
	if match == "exact" && !exact {
		return Media{}, ErrFrameNotFound
	}
	name := fmt.Sprintf("frame-%d.%s", frameTS.Sub(rec.StartTS).Nanoseconds(), format)
	path, _, err := s.derive(ctx, rec, tl, name, func(dst string) []string {
		codec := []string{"-c:v", "mjpeg", "-q:v", "2"}
		if format == "png" {
			codec = []string{"-c:v", "png"}
		}
		args := []string{"-copyts", "-ss", tl.preseek(keyframe.PTS), "-noaccurate_seek", "-i", tl.path,
			"-map", "0:v:0", "-vf", fmt.Sprintf("select='eq(pts\\,%d)'", frame.PTS), "-frames:v", "1"}
		args = append(args, codec...)
		return append(args, "-f", "image2", "-update", "1", dst)
	})
	if err != nil {
		return Media{}, err
	}
	return Media{
		Result: model.MediaResult{
			RecordingID: rec.RecordingID, SensorID: rec.SensorID,
			RequestedStartTS: ts.UTC(), StartTS: frameTS.UTC(),
			ExpiryTS: rec.ExpiryTS,
		},
		Name: name, Path: path, ExactMatch: exact,
	}, nil
}

// Clip returns the media spanning [startTS, startTS+duration).
func (s *Service) Clip(ctx context.Context, id string, start time.Time, duration time.Duration, format string) (Media, error) {
	if format != "mp4" || duration <= 0 {
		return Media{}, ErrInvalidRequest
	}
	rec, err := s.retrievableRecording(ctx, id)
	if err != nil {
		return Media{}, err
	}
	end := start.Add(duration)
	if start.Before(rec.StartTS) || (rec.EndTS != nil && end.After(*rec.EndTS)) || !end.After(start) {
		return Media{}, ErrNotCovered
	}
	release, err := s.acquire()
	if err != nil {
		return Media{}, err
	}
	defer release()
	tl, releaseMedia, err := s.openTimeline(ctx, rec)
	if err != nil {
		return Media{}, err
	}
	defer releaseMedia()
	if rec.State == "recording" {
		hasLaterFrame, err := tl.hasFrameAfter(ctx, end)
		if err != nil {
			return Media{}, err
		}
		if !hasLaterFrame {
			return Media{}, ErrNotCovered
		}
	}
	keyframe, err := tl.keyframeAtOrBefore(ctx, start, nil)
	if err != nil {
		return Media{}, err
	}
	clipStart := tl.wall(keyframe.PTS)
	name := fmt.Sprintf("clip-%d-%d.mp4", clipStart.Sub(rec.StartTS).Nanoseconds(), end.Sub(rec.StartTS).Nanoseconds())
	path, _, err := s.derive(ctx, rec, tl, name, func(dst string) []string {
		seek, length := tl.cut(keyframe.PTS, end)
		return []string{"-ss", seek, "-i", tl.path, "-t", length, "-map", "0:v:0", "-c", "copy",
			"-movflags", "+faststart", "-f", "mp4", dst}
	})
	if err != nil {
		return Media{}, err
	}
	requestedEnd, clipEnd := end.UTC(), end.UTC()
	return Media{
		Result: model.MediaResult{
			RecordingID: rec.RecordingID, SensorID: rec.SensorID,
			RequestedStartTS: start.UTC(), RequestedEndTS: &requestedEnd,
			StartTS: clipStart.UTC(), EndTS: &clipEnd,
			ExpiryTS: rec.ExpiryTS,
		},
		Name: name, Path: path, ExactMatch: clipStart.Equal(start),
	}, nil
}

// OpenMedia serves derived media until its recording expires or is deleted.
func (s *Service) OpenMedia(ctx context.Context, id, name string) (*os.File, MediaFile, error) {
	rec, err := s.recording(ctx, id)
	if err != nil {
		return nil, MediaFile{}, err
	}
	file, err := s.media.OpenDerivedByID(ctx, id, name)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, MediaFile{}, ErrMediaNotFound
	}
	if err != nil {
		return nil, MediaFile{}, err
	}
	contentType := contentTypes[filepath.Ext(name)[1:]]
	return file, MediaFile{ContentType: contentType, ExpiryTS: rec.ExpiryTS}, nil
}

// recording returns a recording that has not expired.
func (s *Service) recording(ctx context.Context, id string) (model.Recording, error) {
	rec, err := s.metadata.GetMetadataByRecordingID(ctx, id)
	if errors.Is(err, storage.ErrNotFound) {
		return model.Recording{}, ErrNotFound
	}
	if err != nil {
		return model.Recording{}, err
	}
	if rec.ExpiryTS != nil && !s.now().Before(*rec.ExpiryTS) {
		return model.Recording{}, ErrNotFound
	}
	return rec, nil
}

func (s *Service) retrievableRecording(ctx context.Context, id string) (model.Recording, error) {
	rec, err := s.recording(ctx, id)
	if err != nil {
		return model.Recording{}, err
	}
	ready := rec.State == "ready" && rec.EndTS != nil
	live := rec.State == "recording" && rec.EndTS == nil
	if !ready && !live {
		return model.Recording{}, ErrNotReady
	}
	return rec, nil
}

func (s *Service) acquire() (func(), error) {
	select {
	case s.slots <- struct{}{}:
		return func() { <-s.slots }, nil
	default:
		return nil, ErrBusy
	}
}

func (s *Service) openTimeline(ctx context.Context, rec model.Recording) (*timeline, func(), error) {
	release, err := s.media.AcquireMedia(ctx, rec.RecordingPath)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil, ErrNotFound
	}
	if err != nil {
		return nil, nil, err
	}
	path, err := s.media.OpenMedia(ctx, rec.RecordingPath)
	if errors.Is(err, fs.ErrNotExist) {
		err = ErrNotFound
	}
	if err != nil {
		release()
		return nil, nil, err
	}
	tl, err := newTimeline(ctx, s.ffprobe, path, rec.StartTS)
	if err != nil {
		release()
		return nil, nil, err
	}
	return tl, release, nil
}

// derive returns the cached file called name, otherwise
// cuts it again with the FFmpeg arguments built for a destination path.
func (s *Service) derive(ctx context.Context, rec model.Recording, tl *timeline, name string, args func(dst string) []string) (string, time.Time, error) {
	path, mtime, err := s.cachedDerived(ctx, rec, name)
	if err != nil || path != "" {
		return path, mtime, err
	}
	path, mtime, err = s.media.PutDerived(ctx, rec.RecordingPath, name, func(dst string) error {
		return runFFmpeg(ctx, s.ffmpeg, tl.path, ffmpegArgs(args(dst))...)
	})
	if errors.Is(err, storage.ErrEmptyMedia) {
		return "", time.Time{}, fmt.Errorf("%w: %w", ErrExtraction, err)
	}
	return path, mtime, err
}

// cachedDerived returns the path and modification time of the file called
// name, or an empty path when it is missing.
func (s *Service) cachedDerived(ctx context.Context, rec model.Recording, name string) (string, time.Time, error) {
	file, err := s.media.OpenDerived(ctx, rec.RecordingPath, name)
	if errors.Is(err, fs.ErrNotExist) {
		return "", time.Time{}, nil
	}
	if err != nil {
		return "", time.Time{}, err
	}
	info, err := file.Stat()
	if err := errors.Join(err, file.Close()); err != nil {
		return "", time.Time{}, err
	}
	return file.Name(), info.ModTime(), nil
}

// ffmpegArgs prefixes the shared FFmpeg flags.
func ffmpegArgs(args []string) []string {
	return append([]string{"-hide_banner", "-loglevel", "error", "-nostdin", "-y"}, args...)
}

func absDuration(d time.Duration) time.Duration {
	if d < 0 {
		return -d
	}
	return d
}
