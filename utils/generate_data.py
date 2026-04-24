"""
Genera datos sintéticos para entrenamiento
Basado en patrones de los datos existentes
"""

import pandas as pd
import random

existing_data = [
    ("Invoice number 001 from March with total amount of $5000 for consulting services", "invoice"),
    ("Lease agreement valid for 12 months starting from 2024-03-01", "contract"),
    ("Email confirmation: Your payment of $3200 has been received successfully", "email"),
    ("Monthly sales report: Total invoiced $50000 in March", "report"),
    ("INVOICE - Vendor: Acme Inc - Description: Licenses - Amount: $2000", "invoice"),
    ("Confidential agreement between company A and company B valid for 3 years", "contract"),
    ("Message: Dear customer your order has been shipped", "email"),
    ("Quarterly analysis of expenses by department and Q2 projections", "report"),
    ("INV-2024-0315 - For professional services - Total due $7500", "invoice"),
    ("Terms and conditions for technology services contracting", "contract"),
    ("Notification of account status change", "email"),
    ("Executive summary of March 2024 operations", "report"),
    ("Electronic invoice - Client: Tech Solutions - Amount: $4200", "invoice"),
    ("Legal document: Power of attorney for legal representation", "contract"),
    ("Your payment has been processed. Reference: PAY-2024-001", "email"),
    ("Dashboard of KPIs: Revenue, expenses, profitability by business line", "report"),
    ("Invoice for software development services rendered in March", "invoice"),
    ("Service agreement and maintenance contract for 24 months", "contract"),
    ("Your subscription has been renewed for the next billing period", "email"),
    ("Financial performance analysis and budget forecast for next quarter", "report"),
]


def generate_invoices(count=50):
    """Generate invoice examples"""
    vendors = ["Acme Inc", "Tech Solutions", "Global Services", "Digital Agency", 
               "Cloud Providers", "Software House", "Consulting Group", "IT Services",
               "Creative Studio", "Data Analytics"]
    
    amounts = [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000]
    dates = ["January", "February", "March", "April", "May", "June", 
             "July", "August", "September", "October", "November", "December"]
    services = ["consulting services", "software licenses", "support services",
                "development services", "maintenance", "analysis", "training",
                "implementation", "hosting", "professional services"]
    
    invoices = []
    for i in range(count):
        vendor = random.choice(vendors)
        amount = random.choice(amounts)
        date = random.choice(dates)
        service = random.choice(services)
        invoice_num = f"{random.randint(100, 9999)}"
        
        templates = [
            f"Invoice number {invoice_num} from {date} for {service} amount ${amount}",
            f"INVOICE - Vendor: {vendor} - Amount: ${amount} - Date: {date}",
            f"INV-2024-{random.randint(1000, 9999)} - {service.capitalize()} - Total: ${amount}",
            f"Bill from {vendor} for {service} totaling ${amount} in {date}",
            f"Payment invoice #{invoice_num} - {service.capitalize()} - ${amount}",
            f"Electronic invoice from {vendor} for ${amount} ({service})",
            f"Factura #{invoice_num} - {vendor} - ${amount} ({date})",
            f"Invoice {invoice_num}: {service} services rendered - Amount due ${amount}",
        ]
        
        invoices.append(random.choice(templates))
    
    return invoices

def generate_contracts(count=50):
    """Generate contract examples"""
    parties = ["Company A", "Company B", "Tech Corp", "Service Provider", 
               "Vendor", "Client", "Partner", "Organization"]
    durations = ["12 months", "24 months", "36 months", "1 year", "2 years", 
                 "3 years", "5 years", "10 years"]
    types = ["service agreement", "maintenance contract", "license agreement",
             "confidential agreement", "employment contract", "lease agreement",
             "partnership agreement", "non-disclosure agreement"]
    
    contracts = []
    for i in range(count):
        party1 = random.choice(parties)
        party2 = random.choice(parties)
        duration = random.choice(durations)
        contract_type = random.choice(types)
        start_date = f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
        
        templates = [
            f"{contract_type.capitalize()} between {party1} and {party2} valid for {duration}",
            f"Legal {contract_type} - Valid from {start_date} for {duration}",
            f"Contract: {contract_type} - Parties: {party1}, {party2} - Duration: {duration}",
            f"{contract_type.capitalize()} between {party1} and {party2}",
            f"Terms and conditions: {contract_type} for {duration} period",
            f"Signed {contract_type} effective {start_date} through {duration}",
            f"Agreement document: {contract_type} valid for {duration}",
            f"Contract ID #{random.randint(10000, 99999)}: {contract_type} - {duration}",
        ]
        
        contracts.append(random.choice(templates))
    
    return contracts

def generate_emails(count=50):
    """Genera emails examples"""
    senders = ["support@company.com", "info@company.com", "noreply@company.com",
               "billing@company.com", "no-reply@service.com", "customer@service.com",
               "notifications@platform.com", "alerts@system.com"]
    
    messages = ["Your payment has been received", "Order confirmed", 
                "Account updated", "Status changed", "Notification",
                "Confirmation", "Receipt", "Update", "Action required",
                "New message", "Subscription renewed", "Reminder"]
    
    emails = []
    for i in range(count):
        sender = random.choice(senders)
        message = random.choice(messages)
        reference = f"REF-{random.randint(100000, 999999)}"
        
        templates = [
            f"Email from {sender}: {message}. Reference: {reference}",
            f"Message: {message}. Details sent to your account.",
            f"{message.capitalize()} - Email notification received",
            f"Your {message.lower()} notification has been sent",
            f"Email alert: {message}. Transaction ID: {reference}",
            f"Automated email: {message} successfully",
            f"You have a new email: {message}",
            f"System notification: {message.capitalize()}",
            f"Hello, {message.capitalize()}. Ref: {reference}",
        ]
        
        emails.append(random.choice(templates))
    return emails

def generate_reports(count=50):
    """Generate reports examples"""
    periods = ["Monthly", "Quarterly", "Annual", "Weekly", "Daily", "Bi-weekly"]
    metrics = ["sales", "revenue", "expenses", "performance", "analytics", 
               "KPIs", "metrics", "statistics", "analysis", "forecast"]
    departments = ["Sales", "Marketing", "Engineering", "Operations", "Finance",
                   "HR", "Support", "Product"]
    
    reports = []
    for i in range(count):
        period = random.choice(periods)
        metric = random.choice(metrics)
        dept = random.choice(departments)
        month = random.choice(["January", "February", "March", "April", "May", "June"])
        year = random.randint(2023, 2024)
        
        templates = [
            f"{period} report: {metric} analysis for {month} {year}",
            f"{dept} {period.lower()} {metric} report - {month} {year}",
            f"Executive summary of {metric} by department",
            f"{period} {metric} dashboard and KPIs",
            f"Analysis of {metric} and {random.choice(metrics)} for {month}",
            f"Report: {dept} performance indicators",
            f"Summary: {metric.capitalize()} overview {month} {year}",
            f"{period} financial analysis and budget forecast",
            f"Performance report: {metric} trends and projections",
            f"Data analytics: {metric} by department and region",
        ]
        
        reports.append(random.choice(templates))
    
    return reports


print("Generating synthetic data...")

invoices = generate_invoices(60)
contracts = generate_contracts(60)
emails = generate_emails(60)
reports = generate_reports(60)

all_data = (
    [(text, "invoice") for text in invoices] +
    [(text, "contract") for text in contracts] +
    [(text, "email") for text in emails] +
    [(text, "report") for text in reports] +
    existing_data  
)

print(f"✅ Generated {len(all_data)} total examples:")
print(f"   Invoices: {len(invoices) + sum(1 for _, l in existing_data if l == 'invoice')}")
print(f"   Contracts: {len(contracts) + sum(1 for _, l in existing_data if l == 'contract')}")
print(f"   Emails: {len(emails) + sum(1 for _, l in existing_data if l == 'email')}")
print(f"   Reports: {len(reports) + sum(1 for _, l in existing_data if l == 'report')}")

# Mezclar
random.shuffle(all_data)

# Guardar
df = pd.DataFrame(all_data, columns=['text', 'label'])
df.to_csv("training_data_expanded.csv", index=False)

print(f"\n✅ Data saved to 'training_data_expanded.csv'")
print(f"\nDistribution:")
print(df['label'].value_counts())