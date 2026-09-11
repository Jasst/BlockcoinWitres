/**
 * UI Enhancements for Dark Messenger
 * Adds micro-interactions, skeleton loaders, and page transitions
 */

// Page transition on load
document.addEventListener('DOMContentLoaded', () => {
  // Add page load animation
  document.body.classList.add('animate-page');
  
  // Remove loading state when page is ready
  setTimeout(() => {
    document.body.classList.remove('page-loading');
    document.body.classList.add('page-loaded');
  }, 100);
  
  // Initialize button ripple effects
  initButtonRipples();
  
  // Initialize skeleton loaders
  initSkeletonLoaders();
  
  // Initialize lazy loading for images
  initLazyLoading();
  
  // Initialize scroll-based navigation effects
  initScrollEffects();
});

// Button ripple effect
function initButtonRipples() {
  const buttons = document.querySelectorAll('.btn-ripple, .btn-primary-oval, .btn-oval');
  
  buttons.forEach(button => {
    button.addEventListener('click', function(e) {
      const rect = this.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      const ripple = document.createElement('span');
      ripple.style.cssText = `
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        width: 100px;
        height: 100px;
        margin-top: -50px;
        margin-left: -50px;
        top: ${y}px;
        left: ${x}px;
        transform: scale(0);
        opacity: 1;
        pointer-events: none;
        animation: buttonRipple 0.5s ease-out;
      `;
      
      this.style.position = 'relative';
      this.style.overflow = 'hidden';
      this.appendChild(ripple);
      
      setTimeout(() => ripple.remove(), 500);
    });
  });
}

// Skeleton loader utility
function initSkeletonLoaders() {
  // Convert loading spinners to skeletons where appropriate
  const loadingElements = document.querySelectorAll('.loading');
  
  loadingElements.forEach(el => {
    if (el.parentElement.classList.contains('conversation-item') || 
        el.parentElement.classList.contains('message-list')) {
      el.innerHTML = '';
      el.classList.add('skeleton-loader');
      
      // Create skeleton elements
      for (let i = 0; i < 3; i++) {
        const skeleton = document.createElement('div');
        skeleton.className = 'skeleton skeleton-message';
        el.appendChild(skeleton);
      }
    }
  });
}

// Lazy loading for images
function initLazyLoading() {
  if ('loading' in HTMLImageElement.prototype) {
    const images = document.querySelectorAll('img[data-src]');
    images.forEach(img => {
      img.src = img.dataset.src;
    });
  } else {
    // Fallback for browsers that don't support native lazy loading
    const imageObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src;
          img.classList.add('loaded');
          observer.unobserve(img);
        }
      });
    });
    
    document.querySelectorAll('img[data-src]').forEach(img => {
      imageObserver.observe(img);
    });
  }
}

// Scroll-based effects for navigation
function initScrollEffects() {
  const bottomNav = document.querySelector('.bottom-nav');
  const messagesContainer = document.querySelector('.messages');
  
  if (bottomNav) {
    window.addEventListener('scroll', () => {
      if (window.scrollY > 50) {
        bottomNav.classList.add('scrolled');
      } else {
        bottomNav.classList.remove('scrolled');
      }
    });
  }
  
  // Smooth scroll to bottom on new message
  if (messagesContainer) {
    const observer = new MutationObserver(() => {
      messagesContainer.scrollTop = messagesContainer.scrollHeight;
    });
    
    observer.observe(messagesContainer, {
      childList: true,
      subtree: true
    });
  }
}

// Message animation helper
function animateMessage(element, isSent = false) {
  element.classList.add('animate-message');
  
  if (isSent) {
    const content = element.querySelector('.content');
    if (content) {
      content.classList.add('animate-sent');
    }
  }
  
  // Clean up animation classes after completion
  setTimeout(() => {
    element.classList.remove('animate-message');
    if (isSent && content) {
      content.classList.remove('animate-sent');
    }
  }, 1000);
}

// Show skeleton while loading
function showSkeleton(container, type = 'messages', count = 3) {
  container.innerHTML = '';
  
  for (let i = 0; i < count; i++) {
    const skeleton = document.createElement('div');
    skeleton.className = `skeleton skeleton-${type}`;
    container.appendChild(skeleton);
  }
}

// Hide skeleton and show content
function hideSkeleton(container, content) {
  container.innerHTML = '';
  container.appendChild(content);
  container.classList.add('animate-fade');
}

// Export functions for use in other modules
window.UIEnhancements = {
  animateMessage,
  showSkeleton,
  hideSkeleton,
  initButtonRipples,
  initLazyLoading
};
