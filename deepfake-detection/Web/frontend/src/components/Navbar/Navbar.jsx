import styles from './Navbar.module.css'

export default function Navbar() {
  return (
    <nav className={styles.nav}>
      <div className={styles.logo}>
        <span className={styles.dot} />
        DeepfakeDetect
      </div>
      <div className={styles.links}>
        <a href="#" className={styles.link}>About</a>
        <a href="#" className={styles.link}>Documentation</a>
      </div>
    </nav>
  )
}