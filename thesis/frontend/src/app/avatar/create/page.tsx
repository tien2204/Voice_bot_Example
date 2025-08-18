"use client";

import { AvatarCreatorComponent } from '@/components/AvatarCreator/AvatarCreatorComponent';
import { useAuth } from '@/contexts/AuthContext';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function AvatarCreatePage() {
  const { isAuthenticated, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push('/login'); // Redirect to login if not authenticated
    }
  }, [isAuthenticated, isLoading, router]);

  if (isLoading || !isAuthenticated) {
    return <div>Loading...</div>;
  }

  return <AvatarCreatorComponent onAvatarExported={(url) => console.log('Avatar Exported on page:', url)} />;
}