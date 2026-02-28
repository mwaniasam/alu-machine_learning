-- Creates a trigger that decreases item quantity after a new order
CREATE TRIGGER decrease_quantity
AFTER INSERT ON orders
FOR EACH ROW
UPDATE items
SET quantity = quantity - NEW.number
WHERE name = NEW.item_name;