import React, { useState, useEffect } from 'react';
import type { QuizResults, WrongAnswerDetail } from '../../types/quiz';

export interface ResultsScreenProps {
  results: QuizResults;
  onRestart: () => void;
}

export const ResultsScreen: React.FC<ResultsScreenProps> = ({
  results,
  onRestart,
}) => {
  const [isMobile, setIsMobile] = useState(false);
  const [showReview, setShowReview] = useState(false);

  useEffect(() => {
    const checkMedia = () => {
      setIsMobile(window.matchMedia('(max-width: 768px)').matches);
    };
    checkMedia();
    const query = window.matchMedia('(max-width: 768px)');
    query.addEventListener('change', checkMedia);
    return () => query.removeEventListener('change', checkMedia);
  }, []);

  const getScoreMessage = (score: number): string => {
    if (score >= 90) return 'Excellent';
    if (score >= 80) return 'Great work';
    if (score >= 70) return 'Good job';
    if (score >= 60) return 'Passing';
    return 'Keep practicing';
  };

  const getScoreVariant = (score: number): 'success' | 'warning' | 'danger' => {
    if (score >= 70) return 'success';
    if (score >= 60) return 'warning';
    return 'danger';
  };

  const variant = getScoreVariant(results.score);

  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    if (minutes > 0) {
      return `${minutes}m ${remainingSeconds}s`;
    }
    return `${remainingSeconds}s`;
  };

  const variantColors = {
    success: { bg: '#ecfdf5', border: '#a7f3d0', text: '#065f46' },
    warning: { bg: '#fffbeb', border: '#fde68a', text: '#92400e' },
    danger: { bg: '#fef2f2', border: '#fecaca', text: '#991b1b' },
  };

  const colors = variantColors[variant];

  const padding = isMobile ? '24px' : '48px';
  const iconSize = isMobile ? '60px' : '80px';
  const titleSize = isMobile ? '32px' : '42px';
  const innerPadding = isMobile ? '20px' : '32px 48px';
  const statPadding = isMobile ? '14px' : '20px';

  return (
    <div style={{
      minHeight: 'calc(100vh - 65px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: padding,
      backgroundColor: '#f8fafc',
    }}>
      <div style={{
        backgroundColor: 'white',
        borderRadius: '12px',
        border: '1px solid #e2e8f0',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        maxWidth: '700px',
        width: '100%',
        overflow: 'hidden',
      }}>
          <div style={{
            padding: isMobile ? '24px' : '40px 48px',
            textAlign: 'center',
            borderBottom: '1px solid #f1f5f9',
          }}>
            <div style={{
              width: iconSize,
              height: iconSize,
              borderRadius: '50%',
              backgroundColor: colors.bg,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 16px',
              fontSize: isMobile ? '28px' : '36px',
              border: `2px solid ${colors.border}`,
            }}>
              {variant === 'success' ? '✓' : variant === 'warning' ? '!' : '×'}
            </div>
            <h1 style={{
              fontSize: titleSize,
              fontWeight: '700',
              color: '#0f172a',
              marginBottom: '8px',
            }}>
              {results.score}%
            </h1>
            <div style={{
              width: '100%',
              maxWidth: '300px',
              height: '8px',
              backgroundColor: '#e2e8f0',
              borderRadius: '4px',
              margin: '0 auto 12px',
              overflow: 'hidden',
            }}>
              <div style={{
                width: `${results.score}%`,
                height: '100%',
                backgroundColor: variant === 'success' ? '#10b981' : variant === 'warning' ? '#f59e0b' : '#ef4444',
                borderRadius: '4px',
                transition: 'width 0.5s ease-out',
              }} />
            </div>
            <p style={{ fontSize: isMobile ? '16px' : '18px', color: '#475569', margin: 0 }}>
              {getScoreMessage(results.score)}
            </p>
          </div>

        <div style={{ padding: innerPadding }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: isMobile ? 'repeat(2, 1fr)' : 'repeat(4, 1fr)',
            gap: isMobile ? '12px' : '16px',
            marginBottom: isMobile ? '24px' : '32px',
          }}>
            <div style={{
              padding: statPadding,
              backgroundColor: '#f8fafc',
              borderRadius: '8px',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: isMobile ? '22px' : '28px', fontWeight: '600', color: '#0f172a' }}>{results.totalQuestions}</div>
              <div style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', marginTop: '4px' }}>Total</div>
            </div>
            <div style={{
              padding: statPadding,
              backgroundColor: '#ecfdf5',
              borderRadius: '8px',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: isMobile ? '22px' : '28px', fontWeight: '600', color: '#10b981' }}>{results.correctAnswers}</div>
              <div style={{ fontSize: '11px', color: '#166534', textTransform: 'uppercase', marginTop: '4px' }}>Correct</div>
            </div>
            <div style={{
              padding: statPadding,
              backgroundColor: '#fef2f2',
              borderRadius: '8px',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: isMobile ? '22px' : '28px', fontWeight: '600', color: '#ef4444' }}>{results.incorrectAnswers}</div>
              <div style={{ fontSize: '11px', color: '#991b1b', textTransform: 'uppercase', marginTop: '4px' }}>Incorrect</div>
            </div>
            <div style={{
              padding: statPadding,
              backgroundColor: '#f8fafc',
              borderRadius: '8px',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: isMobile ? '22px' : '28px', fontWeight: '600', color: '#64748b' }}>{formatTime(results.timeTaken)}</div>
              <div style={{ fontSize: '11px', color: '#64748b', textTransform: 'uppercase', marginTop: '4px' }}>Time</div>
            </div>
          </div>

          <div style={{
            padding: isMobile ? '14px' : '20px',
            backgroundColor: colors.bg,
            border: `1px solid ${colors.border}`,
            borderRadius: '8px',
            marginBottom: isMobile ? '24px' : '32px',
          }}>
            <div style={{ fontSize: isMobile ? '14px' : '15px', fontWeight: '600', color: colors.text, marginBottom: '6px' }}>
              {variant === 'success' ? 'Strong performance' :
               variant === 'warning' ? 'Review recommended' :
               'More study needed'}
            </div>
            <div style={{ fontSize: isMobile ? '13px' : '14px', color: colors.text, opacity: 0.9 }}>
              {variant === 'success'
                ? 'You have a solid grasp of the AWS ML Specialty material.'
                : variant === 'warning'
                ? 'Review the questions you missed and try again.'
                : 'Focus on the core concepts and try the quiz again.'}
            </div>
          </div>

          {results.wrongAnswers && results.wrongAnswers.length > 0 && (
            <button
              onClick={() => setShowReview(!showReview)}
              style={{
                width: '100%',
                padding: isMobile ? '12px' : '14px 24px',
                backgroundColor: showReview ? '#e2e8f0' : '#f8fafc',
                color: '#0f172a',
                fontSize: isMobile ? '13px' : '14px',
                fontWeight: '500',
                borderRadius: '8px',
                cursor: 'pointer',
                border: '1px solid #e2e8f0',
                marginBottom: '12px',
                transition: 'background-color 0.15s ease',
              }}
            >
              {showReview ? '▲ Hide Review' : `▼ Review ${results.wrongAnswers.length} Incorrect Answer${results.wrongAnswers.length > 1 ? 's' : ''}`}
            </button>
          )}

          {showReview && results.wrongAnswers && results.wrongAnswers.length > 0 && (
            <div style={{
              maxHeight: '400px',
              overflowY: 'auto',
              marginBottom: '16px',
              border: '1px solid #e2e8f0',
              borderRadius: '8px',
              backgroundColor: '#fafafa',
            }}>
              {results.wrongAnswers.map((wrong) => (
                <WrongAnswerCard key={wrong.questionId} wrong={wrong} isMobile={isMobile} />
              ))}
            </div>
          )}

          <button
            onClick={onRestart}
            style={{
              width: '100%',
              padding: isMobile ? '14px' : '16px 32px',
              backgroundColor: '#0f172a',
              color: 'white',
              fontSize: isMobile ? '14px' : '15px',
              fontWeight: '500',
              borderRadius: '8px',
              cursor: 'pointer',
              border: 'none',
              transition: 'background-color 0.15s ease',
            }}
          >
            Try Again
          </button>
        </div>
      </div>
    </div>
  );
};

interface WrongAnswerCardProps {
  wrong: WrongAnswerDetail;
  isMobile: boolean;
}

const WrongAnswerCard: React.FC<WrongAnswerCardProps> = ({ wrong, isMobile }) => {
  const padding = isMobile ? '12px' : '16px';

  return (
    <div style={{
      padding: padding,
      borderBottom: '1px solid #e2e8f0',
      backgroundColor: 'white',
    }}>
      <div style={{
        fontSize: isMobile ? '13px' : '14px',
        fontWeight: '600',
        color: '#0f172a',
        marginBottom: '8px',
      }}>
        <span style={{ color: '#ef4444' }}>Q{wrong.questionNumber}:</span> {wrong.question}
      </div>

      <div style={{ marginBottom: '8px' }}>
        <div style={{
          fontSize: '11px',
          color: '#ef4444',
          fontWeight: '600',
          marginBottom: '4px',
          textTransform: 'uppercase',
        }}>
          Your Answer:
        </div>
        {wrong.selectedOptionIds.map((optId) => {
          const option = wrong.options.find((o) => o.id === optId);
          return (
            <div key={optId} style={{
              padding: '8px 12px',
              backgroundColor: '#fef2f2',
              borderRadius: '4px',
              marginBottom: '4px',
              borderLeft: '3px solid #ef4444',
              fontSize: isMobile ? '12px' : '13px',
              color: '#991b1b',
            }}>
              {option?.text}
            </div>
          );
        })}
      </div>

      <div style={{ marginBottom: '8px' }}>
        <div style={{
          fontSize: '11px',
          color: '#10b981',
          fontWeight: '600',
          marginBottom: '4px',
          textTransform: 'uppercase',
        }}>
          Correct Answer:
        </div>
        {wrong.correctOptionIds.map((optId) => {
          const option = wrong.options.find((o) => o.id === optId);
          return (
            <div key={optId} style={{
              padding: '8px 12px',
              backgroundColor: '#ecfdf5',
              borderRadius: '4px',
              marginBottom: '4px',
              borderLeft: '3px solid #10b981',
              fontSize: isMobile ? '12px' : '13px',
              color: '#065f46',
            }}>
              {option?.text}
            </div>
          );
        })}
      </div>

      {wrong.explanation && (
        <div style={{
          padding: '10px 12px',
          backgroundColor: '#f1f5f9',
          borderRadius: '4px',
          fontSize: isMobile ? '12px' : '13px',
          color: '#475569',
          lineHeight: '1.5',
        }}>
          <strong>Explanation:</strong> {wrong.explanation}
        </div>
      )}
    </div>
  );
};

export default ResultsScreen;
