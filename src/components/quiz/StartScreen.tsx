import React, { useState, useEffect } from 'react';

export interface StartScreenProps {
  totalQuestions: number;
  onStart: () => void;
}

export const StartScreen: React.FC<StartScreenProps> = ({
  totalQuestions,
  onStart,
}) => {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkMedia = () => {
      setIsMobile(window.matchMedia('(max-width: 768px)').matches);
    };
    checkMedia();
    const query = window.matchMedia('(max-width: 768px)');
    query.addEventListener('change', checkMedia);
    return () => query.removeEventListener('change', checkMedia);
  }, []);

  const padding = isMobile ? '24px' : '32px 48px';
  const gap = isMobile ? '32px' : '80px';
  const titleSize = isMobile ? '28px' : '36px';
  const paddingVisual = isMobile ? '200px' : '420px';
  const marginBottom = isMobile ? '24px' : '32px';

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: 'calc(100vh - 65px)',
      backgroundColor: '#f8fafc',
      padding: padding,
    }}>
      <div style={{
        display: 'grid',
        gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr',
        gap: gap,
        maxWidth: '1100px',
        width: '100%',
        alignItems: 'center',
      }}>
        <div>
          <h1 style={{
            fontSize: titleSize,
            fontWeight: '700',
            color: '#0f172a',
            marginBottom: '16px',
            lineHeight: '1.15',
          }}>
            AWS ML Specialty<br />Practice Quiz
          </h1>
          <p style={{
            fontSize: isMobile ? '14px' : '16px',
            color: '#475569',
            lineHeight: '1.65',
            marginBottom: marginBottom,
            maxWidth: '480px',
          }}>
            Test your knowledge with {totalQuestions} exam-style questions.
            Review detailed explanations after each answer.
          </p>

          <div style={{
            display: 'flex',
            gap: isMobile ? '24px' : '48px',
            marginBottom: isMobile ? '32px' : '40px',
            flexWrap: 'wrap',
          }}>
            <div>
              <div style={{ fontSize: isMobile ? '24px' : '28px', fontWeight: '600', color: '#0f172a' }}>
                {totalQuestions}
              </div>
              <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase', marginTop: '4px' }}>
                Questions
              </div>
            </div>
            <div>
              <div style={{ fontSize: isMobile ? '24px' : '28px', fontWeight: '600', color: '#0f172a' }}>
                —
              </div>
              <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase', marginTop: '4px' }}>
                Time Limit
              </div>
            </div>
            <div>
              <div style={{ fontSize: isMobile ? '24px' : '28px', fontWeight: '600', color: '#0f172a' }}>
                2
              </div>
              <div style={{ fontSize: '12px', color: '#64748b', textTransform: 'uppercase', marginTop: '4px' }}>
                Types
              </div>
            </div>
          </div>

          <button
            onClick={onStart}
            style={{
              padding: isMobile ? '12px 28px' : '14px 32px',
              backgroundColor: '#0f172a',
              color: 'white',
              fontSize: isMobile ? '14px' : '15px',
              fontWeight: '500',
              borderRadius: '8px',
              border: 'none',
              cursor: 'pointer',
              width: isMobile ? '100%' : 'auto',
            }}
          >
            Start Quiz
          </button>
        </div>

        <div style={{
          backgroundColor: '#e2e8f0',
          borderRadius: '12px',
          height: paddingVisual,
          display: isMobile ? 'none' : 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        </div>
      </div>
    </div>
  );
};

export default StartScreen;
