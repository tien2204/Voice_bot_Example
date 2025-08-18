// "use client";
// import { useAuth } from '@/contexts/AuthContext';
import { redirect } from 'next/navigation';


export default async function Page() {
  redirect('/login');
  return null;
}
