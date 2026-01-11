import React, { useState, useEffect } from 'react';
import type { Question, UserAnswer } from '../../types/quiz';

export interface FeedbackCardProps {
  question: Question;
  userAnswer: UserAnswer;
  onNext: () => void;
  isLastQuestion: boolean;
}

export const FeedbackCard: React.FC<FeedbackCardProps> = ({
  question,
  userAnswer,
  onNext,
  isLastQuestion,
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

  const padding = isMobile ? '16px' : '24px';
  const gap = isMobile ? '24px' : '40px';
  const titleSize = isMobile ? '18px' : '20px';
  const iconSize = isMobile ? '28px' : '32px';
  const actionPadding = isMobile ? '16px' : '20px 24px';

  const isCorrect = userAnswer.isCorrect;
  const correctOptions = question.options.filter((opt) => opt.isCorrect);
  const selectedOptions = question.options.filter((opt) =>
    userAnswer.selectedOptionIds.includes(opt.id)
  );

  return (
    <div style={{ minHeight: 'calc(100vh - 65px)', display: 'flex', flexDirection: 'column' }}>
      <div style={{
        backgroundColor: isCorrect ? '#ecfdf5' : '#fef2f2',
        borderBottom: `1px solid ${isCorrect ? '#a7f3d0' : '#fecaca'}`,
        padding: isMobile ? '12px 16px' : '16px 24px',
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', alignItems: 'center', gap: isMobile ? '12px' : '16px' }}>
          <div style={{
            width: iconSize,
            height: iconSize,
            borderRadius: '50%',
            backgroundColor: isCorrect ? '#10b981' : '#ef4444',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            {isCorrect ? (
              <svg width={isMobile ? "14" : "18"} height={isMobile ? "14" : "18"} viewBox="0 0 20 20" fill="white">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg width={isMobile ? "14" : "18"} height={isMobile ? "14" : "18"} viewBox="0 0 20 20" fill="white">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            )}
          </div>
          <span style={{ fontWeight: '600', fontSize: isMobile ? '14px' : '16px', color: isCorrect ? '#065f46' : '#991b1b' }}>
            {isCorrect ? 'Correct' : 'Incorrect'}
          </span>
        </div>
      </div>

      <div style={{
        flex: 1,
        display: 'grid',
        gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr',
        gap: gap,
        padding: padding,
        maxWidth: '1200px',
        margin: '0 auto',
        width: '100%',
      }}>
        <div>
          <div style={{ marginBottom: isMobile ? '20px' : '32px' }}>
            <span style={{
              display: 'inline-block',
              padding: isMobile ? '4px 10px' : '6px 12px',
              backgroundColor: '#f1f5f9',
              color: '#475569',
              fontSize: isMobile ? '11px' : '12px',
              fontWeight: '500',
              borderRadius: '4px',
              marginBottom: isMobile ? '12px' : '16px',
            }}>
              Question
            </span>
            <h2 style={{
              fontSize: titleSize,
              fontWeight: '500',
              color: '#0f172a',
              lineHeight: '1.55',
              marginBottom: isMobile ? '16px' : '24px',
            }}>
              {question.question}
            </h2>

            {question.explanation && (
              <div style={{
                padding: isMobile ? '14px' : '20px',
                backgroundColor: '#f8fafc',
                borderRadius: '8px',
                border: '1px solid #e2e8f0',
              }}>
                <div style={{ fontSize: isMobile ? '11px' : '12px', fontWeight: '600', color: '#64748b', textTransform: 'uppercase', marginBottom: isMobile ? '8px' : '12px' }}>
                  Explanation
                </div>
                <p style={{ fontSize: isMobile ? '13px' : '14px', color: '#475569', lineHeight: '1.65', margin: 0 }}>
                  {question.explanation}
                </p>
              </div>
            )}
          </div>
        </div>

        <div>
          {!isCorrect && selectedOptions.length > 0 && (
            <div style={{ marginBottom: isMobile ? '16px' : '24px' }}>
              <div style={{ fontSize: isMobile ? '11px' : '12px', fontWeight: '600', color: '#64748b', textTransform: 'uppercase', marginBottom: isMobile ? '8px' : '12px' }}>
                Your Answer
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: isMobile ? '8px' : '10px' }}>
                {selectedOptions.map((option) => (
                  <div
                    key={option.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      padding: isMobile ? '12px' : '14px 16px',
                      backgroundColor: '#fef2f2',
                      border: '1px solid #fecaca',
                      borderRadius: '8px',
                    }}
                  >
                    <div style={{
                      width: '20px',
                      height: '20px',
                      borderRadius: '50%',
                      backgroundColor: '#ef4444',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0,
                    }}>
                      <svg width="12" height="12" viewBox="0 0 20 20" fill="white">
                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <span style={{ fontSize: isMobile ? '13px' : '14px', color: '#7f1d1d' }}>{option.text}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div>
            <div style={{ fontSize: isMobile ? '11px' : '12px', fontWeight: '600', color: '#64748b', textTransform: 'uppercase', marginBottom: isMobile ? '8px' : '12px' }}>
              Correct Answer
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: isMobile ? '8px' : '10px' }}>
              {correctOptions.map((option) => (
                <div
                  key={option.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    padding: isMobile ? '12px' : '14px 16px',
                    backgroundColor: '#ecfdf5',
                    border: '1px solid #a7f3d0',
                    borderRadius: '8px',
                  }}
                >
                  <div style={{
                    width: '20px',
                    height: '20px',
                    borderRadius: '50%',
                    backgroundColor: '#10b981',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                  }}>
                    <svg width="12" height="12" viewBox="0 0 20 20" fill="white">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <span style={{ fontSize: isMobile ? '13px' : '14px', color: '#064e3b' }}>{option.text}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div style={{
        backgroundColor: 'white',
        borderTop: '1px solid #e2e8f0',
        padding: actionPadding,
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', justifyContent: 'flex-end' }}>
          <button
            onClick={onNext}
            style={{
              padding: isMobile ? '12px 24px' : '14px 32px',
              backgroundColor: '#0f172a',
              color: 'white',
              fontSize: isMobile ? '14px' : '15px',
              fontWeight: '500',
              borderRadius: '8px',
              cursor: 'pointer',
              border: 'none',
              transition: 'background-color 0.15s ease',
              width: isMobile ? '100%' : 'auto',
            }}
          >
            {isLastQuestion ? 'View Results' : 'Next Question'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default FeedbackCard;
