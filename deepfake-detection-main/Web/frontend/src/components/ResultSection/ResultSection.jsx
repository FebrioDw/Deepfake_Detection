import styles from './ResultSection.module.css'

function MetricCard({ value, label, color }) {
  return (
    <div className={styles.metricCard}>
      <div className={styles.metricVal} style={color ? { color } : {}}>
        {value}
      </div>
      <div className={styles.metricLbl}>{label}</div>
    </div>
  )
}

function ProgressBar({ label, value }) {
  const barColor = value >= 80 ? '#E24B4A'
                 : value >= 50 ? '#EF9F27'
                 : '#639922'
  return (
    <div className={styles.progressRow}>
      <span className={styles.progressLabel}>{label}</span>
      <div className={styles.progressBg}>
        <div className={styles.progressFill}
             style={{ width: `${value}%`, background: barColor }} />
      </div>
      <span className={styles.progressVal} style={{ color: barColor }}>
        {value}%
      </span>
    </div>
  )
}

export default function ResultSection({ result, onReset }) {
  const { verdict, video_meta, frame_results, gradcam, video_info } = result
  const isFake = verdict.is_deepfake

  return (
    <div className={styles.wrapper}>

      {/* Header verdict */}
      <div className={styles.resultHeader}>
        <h2 className={styles.resultTitle}>Detection results</h2>
        <div className={styles.headerRight}>
          <span className={`${styles.verdictBadge} ${isFake ? styles.fake : styles.real}`}>
            {isFake ? '⚠ ' : '✓ '}
            {verdict.label} · {verdict.confidence}%
          </span>
          <button className={styles.btnReset} onClick={onReset}>
            ↺ New scan
          </button>
        </div>
      </div>

      {/* Model info badge */}
      <div className={styles.modelInfo}>
        Model: <strong>Xception</strong>
        &nbsp;·&nbsp; Accuracy 96.2% &nbsp;·&nbsp; AUC-ROC 0.990
        &nbsp;·&nbsp; Input 299×299px
      </div>

      {/* Metric cards */}
      <div className={styles.metricsGrid}>
        <MetricCard
          value={`${verdict.confidence}%`}
          label="Fake confidence"
          color={isFake ? 'var(--color-fake)' : 'var(--color-real)'}
        />
        <MetricCard value={verdict.frames_analyzed} label="Frames analyzed" />
        <MetricCard
          value={verdict.frames_flagged}
          label="Frames flagged"
          color={verdict.frames_flagged > 0 ? 'var(--color-fake)' : 'var(--color-real)'}
        />
        <MetricCard value={`${verdict.inference_ms}ms`} label="Inference / frame" />
      </div>

      {/* Per-frame scores */}
      <div className={styles.card}>
        <div className={styles.cardLabel}>Per-frame prediction score</div>
        {frame_results.map((fr, i) => (
          <ProgressBar
            key={fr.frame_idx}
            label={`Frame ${i + 1}`}
            value={fr.fake_prob}
          />
        ))}
      </div>

      {/* Extracted frames */}
      <div className={styles.card}>
        <div className={styles.cardLabel}>Extracted frames</div>
        <div className={styles.framesGrid}>
          {frame_results.map((fr, i) => (
            <div key={fr.frame_idx} className={styles.frameItem}>
              {fr.face_b64 ? (
                <img src={fr.face_b64} alt={`Frame ${i+1}`}
                     className={styles.frameImg} />
              ) : (
                <div className={styles.framePlaceholder}>
                  <span className={styles.framePlaceholderText}>
                    Frame {i+1}
                  </span>
                </div>
              )}
              <span className={`${styles.frameBadge}
                ${fr.is_fake ? styles.badgeFake : styles.badgeReal}`}>
                {fr.is_fake ? 'Fake' : 'Real'} {fr.fake_prob}%
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Grad-CAM */}
      {gradcam && gradcam.length > 0 && (
        <div className={styles.card}>
          <div className={styles.cardLabel}>
            Grad-CAM explainability — top flagged frames
          </div>
          <p className={styles.gradcamDesc}>
            Heatmap shows which facial regions Xception focuses on
            when detecting manipulation. Warmer colors indicate
            higher activation regions.
          </p>
          {gradcam.map((gc, i) => (
            <div key={gc.frame_idx} className={styles.gradcamBlock}>
              <div className={styles.gradcamMeta}>
                Frame {gc.frame_idx}
                &nbsp;·&nbsp;
                <span style={{ color: 'var(--color-fake)', fontWeight: 500 }}>
                  {gc.fake_prob}% fake confidence
                </span>
              </div>
              <div className={styles.gradcamGrid}>
                <div className={styles.gradcamItem}>
                  <img src={gc.original_b64} alt="Original"
                       className={styles.gradcamImg} />
                  <div className={styles.gradcamCaption}>Original</div>
                </div>
                <div className={styles.gradcamItem}>
                  <img src={gc.heatmap_b64} alt="Heatmap"
                       className={styles.gradcamImg} />
                  <div className={styles.gradcamCaption}>Grad-CAM heatmap</div>
                </div>
                <div className={styles.gradcamItem}>
                  <img src={gc.overlay_b64} alt="Overlay"
                       className={styles.gradcamImg} />
                  <div className={styles.gradcamCaption}>Overlay</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Video info */}
      <div className={styles.card}>
        <div className={styles.cardLabel}>Video information</div>
        <div className={styles.infoGrid}>
          <div className={styles.infoItem}>
            <span className={styles.infoKey}>Filename</span>
            <span className={styles.infoVal}>{video_info?.filename || '—'}</span>
          </div>
          <div className={styles.infoItem}>
            <span className={styles.infoKey}>Duration</span>
            <span className={styles.infoVal}>{video_meta?.duration}s</span>
          </div>
          <div className={styles.infoItem}>
            <span className={styles.infoKey}>Resolution</span>
            <span className={styles.infoVal}>
              {video_meta?.width} × {video_meta?.height}
            </span>
          </div>
          <div className={styles.infoItem}>
            <span className={styles.infoKey}>Frame rate</span>
            <span className={styles.infoVal}>{video_meta?.fps} fps</span>
          </div>
          <div className={styles.infoItem}>
            <span className={styles.infoKey}>Total frames</span>
            <span className={styles.infoVal}>{video_meta?.total_frames}</span>
          </div>
          <div className={styles.infoItem}>
            <span className={styles.infoKey}>File size</span>
            <span className={styles.infoVal}>{video_info?.size_kb} KB</span>
          </div>
          <div className={styles.infoItem}>
            <span className={styles.infoKey}>Detection model</span>
            <span className={styles.infoVal}
                  style={{ color: 'var(--color-primary)', fontWeight: 500 }}>
              Xception
            </span>
          </div>
          <div className={styles.infoItem}>
            <span className={styles.infoKey}>Input size</span>
            <span className={styles.infoVal}>
              {video_info?.input_size || '299×299'}
            </span>
          </div>
        </div>
      </div>

    </div>
  )
}