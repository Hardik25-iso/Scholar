import { Navigate, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "./AuthContext";

/** Gate for authenticated routes: redirect to /login until /me confirms a user. */
export default function ProtectedRoute() {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-paper">
        <span className="font-mono text-xs uppercase tracking-[0.18em] text-faint">
          checking session…
        </span>
      </div>
    );
  }
  // Where they were going travels with the redirect, so signing in returns them
  // to it. Without this an invitation link (/join?token=…) is destroyed by the
  // very login it requires — the token is in the URL being discarded.
  return user ? <Outlet /> : <Navigate to="/login" state={{ from: location }} replace />;
}
