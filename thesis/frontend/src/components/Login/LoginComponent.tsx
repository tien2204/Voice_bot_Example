import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '100vh',
    padding: '20px',
    backgroundColor: '#f4f7f6',
  },
  form: {
    padding: '30px',
    borderRadius: '8px',
    backgroundColor: 'white',
    boxShadow: '0 4px 15px rgba(0, 0, 0, 0.1)',
    width: '100%',
    maxWidth: '400px',
  },
  title: {
    marginBottom: '25px',
    color: '#333',
    fontSize: '24px',
    textAlign: 'center',
  },
  inputGroup: {
    marginBottom: '20px',
  },
  label: {
    display: 'block',
    marginBottom: '8px',
    color: '#555',
    fontSize: '14px',
  },
  input: {
    width: '100%',
    padding: '12px',
    border: '1px solid #ddd',
    borderRadius: '4px',
    boxSizing: 'border-box',
    fontSize: '16px',
  },
  button: {
    width: '100%',
    padding: '12px',
    border: 'none',
    borderRadius: '4px',
    backgroundColor: '#007bff',
    color: 'white',
    fontSize: '16px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  error: {
    color: 'red',
    marginTop: '10px',
    textAlign: 'center',
  }
};

export function LoginComponent() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { login, error } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await login(email, password);
  };

  return (
    <div style={styles.container}>
      <form onSubmit={handleSubmit} style={styles.form}>
        <h2 style={styles.title}>Login</h2>
        <div style={styles.inputGroup}><label htmlFor="email" style={styles.label}>Email:</label><input type="email" id="email" value={email} onChange={(e) => setEmail(e.target.value)} required style={styles.input} /></div>
        <div style={styles.inputGroup}><label htmlFor="password" style={styles.label}>Password:</label><input type="password" id="password" value={password} onChange={(e) => setPassword(e.target.value)} required style={styles.input} /></div>
        <button type="submit" style={styles.button}>Login</button>
        {error && <p style={styles.error}>{error}</p>}
      </form>
    </div>
  );
}