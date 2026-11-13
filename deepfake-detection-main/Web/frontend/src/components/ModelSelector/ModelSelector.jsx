import styles from './ModelSelector.module.css'

export default function ModelSelector({ models, selected, onSelect }) {
  return (
    <div className={styles.card}>
      <div className={styles.grid}>
        {models.map(m => (
          <button
            key={m.id}
            className={`${styles.modelCard} ${selected === m.id ? styles.active : ''}`}
            onClick={() => onSelect(m.id)}
          >
            <div className={styles.name}>{m.name}</div>
            <div className={styles.desc}>{m.desc}</div>
            <div className={styles.params}>{m.params}</div>
          </button>
        ))}
      </div>
    </div>
  )
}