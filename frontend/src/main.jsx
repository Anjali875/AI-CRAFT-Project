import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ResultProvider } from './ResultContext'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <ResultProvider>
        <App />
      </ResultProvider>
    </BrowserRouter>
  </StrictMode>,
)