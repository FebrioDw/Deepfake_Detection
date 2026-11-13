import { useState, useRef, useCallback } from 'react'
import styles from './UploadSection.module.css'

function formatSize(bytes) {
  if (bytes < 1024)       return `${bytes} B`
  if (bytes < 1024*1024)  return `${(bytes/1024).toFixed(1)} KB`
  return `${(bytes/(1024*1024)).toFixed(1)} MB`
}

export default function UploadSection({ videoFile, onFileSelect }) {
  const [dragging, setDragging] = useState(false)
  const [preview,  setPreview]  = useState(null)
  const inputRef = useRef(null)

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('video/')) return
    onFileSelect(file)

    const url = URL.createObjectURL(file)
    setPreview(url)
  }, [onFileSelect])

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }

  const onInputChange = (e) => {
    const file = e.target.files[0]
    handleFile(file)
  }

  const onRemove = () => {
    onFileSelect(null)
    setPreview(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  if (videoFile && preview) {
    return (
      <div className={styles.card}>
        <div className={styles.previewRow}>
          <video
            src={preview}
            className={styles.videoPreview}
            controls
            muted
          />
          <div className={styles.previewInfo}>
            <p className={styles.fileName}>{videoFile.name}</p>
            <div className={styles.tags}>
              <span className={styles.tag}>{formatSize(videoFile.size)}</span>
              <span className={styles.tag}>{videoFile.type.split('/')[1].toUpperCase()}</span>
            </div>
            <button className={styles.btnRemove} onClick={onRemove}>
              ✕ Remove video
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.card}>
      <div
        className={`${styles.dropZone} ${dragging ? styles.dragging : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <div className={styles.uploadIcon}>↑</div>
        <p className={styles.uploadText}>
          Drag & drop your video here, or <span className={styles.browse}>browse</span>
        </p>
        <p className={styles.uploadHint}>Supports MP4, AVI, MOV · Max 500 MB</p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className={styles.hiddenInput}
        onChange={onInputChange}
      />
    </div>
  )
}