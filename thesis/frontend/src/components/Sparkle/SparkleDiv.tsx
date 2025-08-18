import React from 'react';
import styles from './SparkleDiv.module.css';

// SVG component for the sparkle shape
const SparkleIcon = () => (
  <svg
    xmlnsXlink="http://www.w3.org/1999/xlink"
    viewBox="0 0 784.11 815.53"
    style={{
      shapeRendering: 'geometricPrecision',
      textRendering: 'geometricPrecision',
      imageRendering: 'auto', // Changed from optimizeQuality for broader compatibility if needed
      fillRule: 'evenodd',
      clipRule: 'evenodd',
    }}
    version="1.1"
    xmlSpace="preserve"
    xmlns="http://www.w3.org/2000/svg"
  >
    <defs></defs>
    <g id="Layer_x0020_1">
      <metadata id="CorelCorpID_0Corel-Layer"></metadata>
      <path
        d="M392.05 0c-20.9,210.08 -184.06,378.41 -392.05,407.78 207.96,29.37 371.12,197.68 392.05,407.74 20.93,-210.06 184.09,-378.37 392.05,-407.74 -207.98,-29.38 -371.16,-197.69 -392.06,-407.78z"
        className={styles.fil0} // Use CSS module class for fill
      ></path>
    </g>
  </svg>
);

interface SparkleDivProps {
  children: React.ReactNode;
}

export const SparkleDiv: React.FC<SparkleDivProps> = ({ children }) => {
  return (
    <span className={styles.sparkleWrapper}>
      <div>{children}</div>
      <div className={`${styles.star} ${styles.star1}`}><SparkleIcon /></div>
      <div className={`${styles.star} ${styles.star2}`}><SparkleIcon /></div>
      <div className={`${styles.star} ${styles.star3}`}><SparkleIcon /></div>
      <div className={`${styles.star} ${styles.star4}`}><SparkleIcon /></div>
      <div className={`${styles.star} ${styles.star5}`}><SparkleIcon /></div>
      <div className={`${styles.star} ${styles.star6}`}><SparkleIcon /></div>
    </span>
  );
};
