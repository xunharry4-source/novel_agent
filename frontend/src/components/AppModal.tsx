import { Modal, type ModalProps } from '@mantine/core';

/**
 * Project-wide Modal defaults.
 * MainLayout scrolls inside MUI <main>; locking body scroll conflicts with
 * index.css / React StrictMode and can leave a stuck overlay (black screen).
 */
export function AppModal({ lockScroll = false, ...props }: ModalProps) {
  return <Modal lockScroll={lockScroll} {...props} />;
}
