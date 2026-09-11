/**
 * Skeleton Loader Helper - replaces spinners with skeleton screens
 */
window.SkeletonLoader = {
  // Create a skeleton for conversation list
  createConversationSkeletons(count = 5) {
    let html = '';
    for (let i = 0; i < count; i++) {
      html += `
        <div class="conversation-item" style="padding: 12px 16px; gap: 12px;">
          <div class="skeleton skeleton-avatar"></div>
          <div style="flex: 1; min-width: 0;">
            <div class="skeleton skeleton-title" style="width: ${60 + Math.random() * 30}%"></div>
            <div class="skeleton skeleton-subtitle" style="width: ${40 + Math.random() * 40}%"></div>
          </div>
        </div>`;
    }
    return html;
  },

  // Create skeleton for chat messages
  createMessageSkeletons(count = 3) {
    let html = '';
    for (let i = 0; i < count; i++) {
      const isSent = Math.random() > 0.5;
      html += `
        <div class="message ${isSent ? 'sent' : ''}" style="justify-content: ${isSent ? 'flex-end' : 'flex-start'}">
          ${!isSent ? '<div class="skeleton skeleton-avatar"></div>' : ''}
          <div class="skeleton skeleton-chat-bubble" style="width: ${100 + Math.random() * 150}px"></div>
        </div>`;
    }
    return html;
  },

  // Create skeleton for contacts
  createContactSkeletons(count = 8) {
    let html = '';
    for (let i = 0; i < count; i++) {
      html += `
        <div class="contact-row" style="padding: 12px 16px; gap: 12px;">
          <div class="skeleton skeleton-avatar"></div>
          <div style="flex: 1;">
            <div class="skeleton skeleton-text medium"></div>
            <div class="skeleton skeleton-text short"></div>
          </div>
        </div>`;
    }
    return html;
  },

  // Replace loading spinner with skeletons
  replaceLoadingWithSkeleton(containerId, type = 'conversations', count = 5) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    let skeletonHtml = '';
    switch(type) {
      case 'conversations':
        skeletonHtml = this.createConversationSkeletons(count);
        break;
      case 'messages':
        skeletonHtml = this.createMessageSkeletons(count);
        break;
      case 'contacts':
        skeletonHtml = this.createContactSkeletons(count);
        break;
      default:
        skeletonHtml = `<div class="skeleton skeleton-text long"></div>
                        <div class="skeleton skeleton-text medium"></div>
                        <div class="skeleton skeleton-text short"></div>`;
    }
    
    container.innerHTML = skeletonHtml;
  },

  // Fade in content after loading
  fadeInContent(containerId, duration = 300) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.style.opacity = '0';
    container.style.transition = `opacity ${duration}ms ease-out`;
    
    requestAnimationFrame(() => {
      container.style.opacity = '1';
    });
  }
};

// Auto-initialize lazy loading for images
document.addEventListener('DOMContentLoaded', () => {
  const lazyImages = document.querySelectorAll('img[lazy="true"], img[data-src]');
  
  if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          const src = img.dataset.src || img.src;
          
          img.classList.add('loading');
          img.onload = () => {
            img.classList.remove('loading');
            img.classList.add('loaded');
          };
          img.src = src;
          
          imageObserver.unobserve(img);
        }
      });
    });
    
    lazyImages.forEach(img => imageObserver.observe(img));
  } else {
    // Fallback for browsers without IntersectionObserver
    lazyImages.forEach(img => {
      const src = img.dataset.src || img.src;
      img.src = src;
    });
  }
});
