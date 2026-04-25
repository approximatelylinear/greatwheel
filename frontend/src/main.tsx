import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { Observer } from './observer/Observer';
import './styles.css';

const root = document.getElementById('root');
if (!root) throw new Error('missing #root');
const url = new URL(window.location.href);
const observe = url.searchParams.get('observe');
createRoot(root).render(
  <StrictMode>
    {observe ? <Observer sessionId={observe} /> : <App />}
  </StrictMode>,
);
