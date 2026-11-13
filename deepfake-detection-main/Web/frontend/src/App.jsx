import { useState } from 'react'
import Navbar from './components/Navbar/Navbar'
import UploadSection from './components/UploadSection/UploadSection'
import ResultSection from './components/ResultSection/ResultSection'
import styles from './App.module.css'

const API_BASE = 'http://localhost:8000'
const SELECTED_MODEL = 'xception'  // Model terbaik — Acc 96.2% AUC 0.990

export default function App() {
  const [videoFile, setVideoFile] = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [result,    setResult]    = useState(null)
  const [error,     setError]     = useState(null)

  const handleDetect = async () => {
    if (!videoFile) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('video',    videoFile)
      formData.append('model_id', SELECTED_MODEL)

      const res = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        body:   formData,
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Server error')
      }

      const data = await res.json()
      setResult(data)

      setTimeout(() => {
        document.getElementById('result-section')
          ?.scrollIntoView({ behavior: 'smooth' })
      }, 100)

    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setVideoFile(null)
    setResult(null)
    setError(null)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className={styles.app}>
      <Navbar />

      <div className={styles.hero}>
        <h1 className={styles.heroTitle}>Deepfake Video Detector</h1>
        <p className={styles.heroDesc}>
          Upload a video and get frame-level deepfake analysis
          with Grad-CAM visual explainability
        </p>
        <div className={styles.modelBadge}>
          Powered by <strong>Xception</strong>
          &nbsp;·&nbsp; Accuracy 96.2%
          &nbsp;·&nbsp; AUC-ROC 0.990
          &nbsp;·&nbsp; FaceForensics++ C23
        </div>
      </div>

      <main className={styles.main}>

        <div className={styles.stepLabel}>Upload video</div>
        <UploadSection
          videoFile={videoFile}
          onFileSelect={setVideoFile}
        />

        <button
          className={styles.btnRun}
          onClick={handleDetect}
          disabled={!videoFile || loading}
        >
          {loading ? (
            <><span className={styles.spinner} /> Analyzing video...</>
          ) : (
            <>&#9658;&nbsp;&nbsp;Run detection</>
          )}
        </button>

        {error && (
          <div className={styles.errorBox}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div id="result-section">
            <hr className={styles.divider} />
            <ResultSection result={result} onReset={handleReset} />
          </div>
        )}
      </main>

      <footer className={styles.footer}>
        <p>DeepfakeDetect · Xception Model · FF++ C23 Deepfakes Dataset</p>
      </footer>
    </div>
  )
}