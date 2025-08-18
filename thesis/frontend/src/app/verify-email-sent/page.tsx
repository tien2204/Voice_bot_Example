"use client";

import { useAuth } from "@/contexts/AuthContext";
// import Image from "next/image"; // Not used in this component currently
// import Link from "next/link"; // Replaced Logout Link with a button
import { useRouter } from "next/navigation";
import { useState, useEffect } from 'react';

export default function VerifyEmailSentPage() {
  const { user, logout, requestVerificationEmail, error: authError, clearError: clearAuthError } = useAuth();
  const router = useRouter();
  const [resendStatusMessage, setResendStatusMessage] = useState<string | null>(null);
  const [isResending, setIsResending] = useState<boolean>(false);

  useEffect(() => {
    // Clear any global auth error when the component mounts or unmounts
    clearAuthError();
    return () => {
      clearAuthError();
    };
  }, [clearAuthError]);

  const handleResendVerificationEmail = async () => {
    if (!user?.email) {
      setResendStatusMessage("User email not found. Please log in again.");
      return;
    }
    setIsResending(true);
    setResendStatusMessage(null);
    clearAuthError(); // Clear previous auth errors before new request

    const result = await requestVerificationEmail(user.email);
    setResendStatusMessage(result.message || (result.success ? "Email sent." : "Failed to send email."));
    setIsResending(false);
  };

  const handleLogout = () => {
    logout();
    router.push("/login");
  };

  return (
    <div className="flex flex-col items-center justify-center bg-gray-900 p-4 flex-grow text-gray-100">
      <div className="w-full max-w-md p-8 space-y-6 bg-gray-800 shadow-xl rounded-xl text-center">
        <div className="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-sky-100">
          <svg className="h-10 w-10 text-sky-600" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" aria-hidden="true">
            <path strokeLinecap="round" strokeLinejoin="round" d="M21.75 9v.906a2.25 2.25 0 01-1.183 1.981l-6.478 3.488M2.25 9v.906a2.25 2.25 0 001.183 1.981l6.478 3.488m8.839 2.51l-4.66-2.51m0 0l-10.43-5.622A2.25 2.25 0 012.25 9V7.5a2.25 2.25 0 012.25-2.25h15A2.25 2.25 0 0121.75 7.5V9" />
          </svg>
        </div>
        <h1 className="mt-4 text-3xl font-extrabold">
          Verify Your Email
        </h1>
        <p className="mt-2 text-gray-300">
          Thanks for registering! We've sent a verification link to {user?.email ? <strong>{user.email}</strong> : "your email address"}.
          Please check your inbox (and spam folder) and click the link to activate your account.
        </p>
        <p className="mt-2 text-sm text-gray-400">
          If you haven't received the email or the link has expired, you can request a new one.
        </p>
        
        <div className="mt-6 space-y-4">
          <button
            onClick={handleResendVerificationEmail}
            disabled={isResending || !user?.email}
            className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium  text-gray-300 bg-gray-700 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isResending ? "Sending..." : "Resend Verification Email"}
          </button>

          {resendStatusMessage && (
            <p className={`mt-2 text-sm ${authError && !resendStatusMessage.toLowerCase().includes("success") ? "text-red-400" : "text-green-400"}`}>
              {resendStatusMessage}
            </p>
          )}
          {/* Display general errors from AuthContext if not already handled by resendStatusMessage */}
          {authError && !resendStatusMessage && (
             <p className="mt-2 text-sm text-red-400">{authError}</p>
          )}

          <button 
            onClick={handleLogout}
            className="w-full flex justify-center py-3 px-4 border border-gray-600 rounded-lg shadow-sm text-sm font-medium text-white bg-sky-500 hover:bg-sky-600  focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-indigo-500"
          >
            Logout
          </button>
        </div>
      </div>
    </div>
  );
}